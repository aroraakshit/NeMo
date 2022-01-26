
import cgi
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

import torch
from hydra.utils import instantiate
from omegaconf import MISSING, DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch import nn

from nemo.collections.tts.helpers.helpers import get_mask_from_lengths
from nemo.collections.tts.models.base import SpectrogramGenerator
from nemo.collections.tts.modules.flowtron_modules import ARStep, ARBackStep
from nemo.core.classes.common import typecheck

@dataclass
class FlowtronConfig:
    encoder: Dict[Any, Any] = MISSING
    melencoder: Dict[Any, Any] = MISSING
    gaussianmixture: Dict[Any, Any] = MISSING
    train_ds: Optional[Dict[Any, Any]] = None
    validation_ds: Optional[Dict[Any, Any]] = None
    dummy_speaker_embedding: bool = MISSING
    speakeremb: Dict[Any, Any] = MISSING
    textemb: Dict[Any, Any] = MISSING
    n_flows: int = MISSING
    n_components: int = MISSING
    use_gate_layer: bool = MISSING
    n_hidden: int = MISSING
    n_attn_channels: int = MISSING
    n_lstm_layers: int = MISSING
    use_cumm_attention: bool = MISSING
    n_mel_channels: int = MISSING
    n_speaker_dim: int = MISSING

class FlowtronModel(SpectrogramGenerator):
    """Flowtron Model that is used to generate mel spectrograms from text"""

    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        super().__init__(cfg=cfg, trainer=trainer)

        # ModelPT ensures that cfg is a DictConfig, but do this second check in case ModelPT changes
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        elif not isinstance(cfg, DictConfig):
            raise ValueError(f"cfg was type: {type(cfg)}. Expected either a dict or a DictConfig")

        # Ensure passed cfg is compliant with schema
        schema = OmegaConf.structured(FlowtronConfig)
        OmegaConf.merge(cfg, schema)

        self.speaker_embedding = torch.nn.Embedding(
            self._cfg.speakeremb.n_speakers, 
            self._cfg.speakeremb.n_speaker_dim
            )
        self.embedding = torch.nn.Embedding(
            self._cfg.textemb.n_text, 
            self._cfg.textemb.n_text_dim
            )
        
        self.flows = torch.nn.ModuleList()
        self.encoder = instantiate(self._cfg.encoder)
        self.dummy_speaker_embedding = self._cfg.dummy_speaker_embedding

        if self._cfg.n_components > 1:
            self.mel_encoder = instantiate(self._cfg.melencoder)
            self.gaussian_mixture = instantiate(self._cfg.gaussianmixture)

        for i in range(self._cfg.n_flows):
            add_gate = True if (i == (self._cfg.n_flows-1) and self._cfg.use_gate_layer) else False
            if i % 2 == 0:
                self.flows.append(ARStep(
                    self._cfg.n_mel_channels, 
                    self._cfg.n_speaker_dim,
                    self._cfg.n_text_dim,
                    self._cfg.n_mel_channels + self._cfg.n_speaker_dim,
                    self._cfg.n_hidden, self._cfg.n_attn_channels,
                    self._cfg.n_lstm_layers, add_gate,
                    self._cfg.use_cumm_attention
                    )
                )
            else:
                self.flows.append(ARBackStep(
                    self._cfg.n_mel_channels, 
                    self._cfg.n_speaker_dim,
                    self._cfg.n_text_dim,
                    self._cfg.n_mel_channels + self._cfg.n_speaker_dim,
                    self._cfg.n_hidden, self._cfg.n_attn_channels,
                    self._cfg.n_lstm_layers, add_gate,
                    self._cfg.use_cumm_attention
                    )
                )
    
    def forward(self, mel, speaker_ids, text, in_lens, out_lens,
                attn_prior=None):
        speaker_ids = speaker_ids*0 if self.dummy_speaker_embedding else speaker_ids
        speaker_vecs = self.speaker_embedding(speaker_ids)
        text = self.embedding(text).transpose(1, 2)
        text = self.encoder(text, in_lens)

        mean, log_var, prob = None, None, None
        if hasattr(self, 'gaussian_mixture'):
            mel_embedding = self.mel_encoder(mel, out_lens)
            mean, log_var, prob = self.gaussian_mixture(
                mel_embedding, mel_embedding.size(0))

        text = text.transpose(0, 1)
        mel = mel.permute(2, 0, 1)

        encoder_outputs = torch.cat(
            [text, speaker_vecs.expand(text.size(0), -1, -1)], 2)
        log_s_list = []
        attns_list = []
        attns_logprob_list = []
        mask = ~get_mask_from_lengths(in_lens)[..., None]
        for i, flow in enumerate(self.flows):
            mel, log_s, gate, attn_out, attn_logprob_out = flow(
                mel, encoder_outputs, mask, out_lens, attn_prior)
            log_s_list.append(log_s)
            attns_list.append(attn_out)
            attns_logprob_list.append(attn_logprob_out)
        return (mel, log_s_list, gate, attns_list,
                attns_logprob_list, mean, log_var, prob) 

    def infer(self, residual, speaker_ids, text, temperature=1.0,
              gate_threshold=0.5, attns=None, attn_prior=None):
        """Inference function. Inverse of the forward pass

        Args:
            residual: 1 x 80 x N_residual tensor of sampled z values
            speaker_ids: 1 x 1 tensor of integral speaker ids (should be a single value)
            text (torch.int64): 1 x N_text tensor holding text-token ids

        Returns:
            residual: input residual after flow transformation. Technically the mel spectrogram values
            attention_weights: attention weights predicted by each flow step for mel-text alignment
        """

        speaker_ids = speaker_ids*0 if self.dummy_speaker_embedding else speaker_ids
        speaker_vecs = self.speaker_embedding(speaker_ids)
        text = self.embedding(text).transpose(1, 2)
        text = self.encoder.infer(text)
        text = text.transpose(0, 1)
        encoder_outputs = torch.cat(
            [text, speaker_vecs.expand(text.size(0), -1, -1)], 2)
        residual = residual.permute(2, 0, 1)
        attention_weights = []
        for i, flow in enumerate(reversed(self.flows)):
            attn = None if attns is None else reversed(attns)[i]
            self.set_temperature_and_gate(flow, temperature, gate_threshold)
            residual, attention_weight = flow.infer(
                residual, encoder_outputs, attn, attn_prior=attn_prior)
            attention_weights.append(attention_weight)
        return residual.permute(1, 2, 0), attention_weights

    def test_invertibility(self, residual, speaker_ids, text, temperature=1.0,
                           gate_threshold=0.5, attns=None):
        """Model invertibility check. Call this the same way you would call self.infer()

        Args:
            residual: 1 x 80 x N_residual tensor of sampled z values
            speaker_ids: 1 x 1 tensor of integral speaker ids (should be a single value)
            text (torch.int64): 1 x N_text tensor holding text-token ids

        Returns:
            error: should be in the order of 1e-5 or less, or there may be an invertibility bug
        """
        mel, attn_weights = self.infer(residual, speaker_ids, text)
        in_lens = torch.LongTensor([text.shape[1]]).cuda()
        residual_recon, log_s_list, gate, _, _, _, _ = self.forward(mel,
                                                                    speaker_ids, text,
                                                                    in_lens, None)
        residual_permuted = residual.permute(2, 0, 1)
        if len(self.flows) % 2 == 0:
            residual_permuted = torch.flip(residual_permuted, (0,))
            residual_recon = torch.flip(residual_recon, (0,))
        error = (residual_recon - residual_permuted[0:residual_recon.shape[0]]).abs().mean()
        return error

    @staticmethod
    def set_temperature_and_gate(flow, temperature, gate_threshold):
        flow = flow.ar_step if hasattr(flow, "ar_step") else flow
        flow.attention_layer.temperature = temperature
        if hasattr(flow, 'gate_layer'):
            flow.gate_threshold = gate_threshold
