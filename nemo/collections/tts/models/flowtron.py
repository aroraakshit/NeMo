
import cgi
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

import torch
from hydra.utils import instantiate
from omegaconf import MISSING, DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch import nn
import random

from nemo.collections.tts.helpers.helpers import get_mask_from_lengths, plot_alignment_to_numpy, plot_gate_outputs_to_numpy
from nemo.collections.tts.models.base import SpectrogramGenerator
from nemo.core.classes.common import PretrainedModelInfo
from nemo.collections.tts.losses.flowtronloss import FlowtronLoss
from nemo.collections.tts.modules.flowtron_submodules import ARStep, ARBackStep, RAdam
from nemo.core.neural_types.neural_type import NeuralType
from pytorch_lightning.loggers import LoggerCollection, TensorBoardLogger
from nemo.core.neural_types.elements import (
    EmbeddedTextType,
    TokenIndex,
    LengthsType,
    LogitsType,
    ProbsType,
    VoidType,
    MelSpectrogramType,
    LogprobsType
)
from nemo.core.classes.common import typecheck

@dataclass
class FlowtronConfig:
    encoder: Dict[Any, Any] = MISSING
    melencoder: Dict[Any, Any] = MISSING
    gaussianmixture: Dict[Any, Any] = MISSING
    flowtronloss: Dict[Any, Any] = MISSING
    arstep: Dict[Any, Any] = MISSING
    trainparams: Dict[Any, Any] = MISSING
    speakeremb: Dict[Any, Any] = MISSING
    textemb: Dict[Any, Any] = MISSING
    train_ds: Optional[Dict[Any, Any]] = None
    validation_ds: Optional[Dict[Any, Any]] = None
    dummy_speaker_embedding: bool = MISSING
    n_flows: int = MISSING
    n_components: int = MISSING
    use_gate_layer: bool = MISSING
    n_speaker_dim: int = MISSING
    optim_algo: str = MISSING
    learning_rate: float = MISSING
    weight_decay: float = MISSING
    finetune_layers: List = MISSING
    seed: int = MISSING

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
                self.flows.append(ARStep(add_gate=add_gate, **self._cfg.arstep))
            else:
                self.flows.append(ARBackStep(add_gate=add_gate, **self._cfg.arstep))

        torch.manual_seed(self._cfg.seed)
        torch.cuda.manual_seed(self._cfg.seed)

        self.criterion = FlowtronLoss(gm_loss=bool(self._cfg.n_components), **self._cfg.flowtronloss)

        if len(self._cfg.finetune_layers):
            for name, param in self.named_parameters():
                if name in self._cfg.finetune_layers:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        self.iteration = 0
        self.apply_ctc = False
    
    @property
    def tb_logger(self):
        if self._tb_logger is None:
            if self.logger is None and self.logger.experiment is None:
                return None
            tb_logger = self.logger.experiment
            if isinstance(self.logger, LoggerCollection):
                for logger in self.logger:
                    if isinstance(logger, TensorBoardLogger):
                        tb_logger = logger.experiment
                        break
            self._tb_logger = tb_logger
        return self._tb_logger

    def parse(self, str_input: str, **kwargs) -> 'torch.tensor':
        trainset = instantiate(self._cfg.train_ds.dataset)
        self.speaker_vecs = trainset.get_speaker_id(kwargs['speaker_id']).cuda()
        text = trainset.get_text(str_input).cuda()
        text_tensor = text.unsqueeze_(0)
        return text_tensor

    @property
    def input_types(self):
        return {
            "mel": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
            "speaker_ids": NeuralType(('B'), LengthsType()),
            "text": NeuralType(('B', 'T_text'), TokenIndex()),
            "in_lens": NeuralType(('B'), LengthsType()),
            "out_lens": NeuralType(('B'), LengthsType()),
            "attn_prior": NeuralType(('B', 'T_spec', 'T_text'), ProbsType(), optional=True),
        }

    @property
    def output_types(self):
        return {
            "mel": NeuralType(('T', 'B', 'D'), MelSpectrogramType()),
            "log_s_list": [NeuralType(('T', 'B', 'D'), VoidType())],
            "gate": NeuralType(('T', 'B', 'D'), LogitsType()),
            "attns_list": [NeuralType(('B', 'T', 'T_text'), VoidType())],
            "attns_logprob_list": [NeuralType(('B', 'T', 'T_text'), LogprobsType())],
            "mean": NeuralType(('B'), LengthsType()),
            "log_var": NeuralType(('B'), LengthsType()), 
            "prob": NeuralType(('B'), ProbsType()),
        }

    @typecheck()
    def forward(self, *, mel, speaker_ids, text, in_lens, out_lens,
                attn_prior=None):
        speaker_ids = speaker_ids*0 if self.dummy_speaker_embedding else speaker_ids
        speaker_vecs = self.speaker_embedding(speaker_ids)
        text = self.embedding(text).transpose(1, 2)
        text = self.encoder(x=text, in_lens=in_lens)

        mean, log_var, prob = None, None, None
        if hasattr(self, 'gaussian_mixture'):
            mel_embedding = self.mel_encoder(x=mel, lens=out_lens)
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
                mel=mel, text=encoder_outputs, mask=mask, out_lens=out_lens, attn_prior=attn_prior)
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
        residual_recon, log_s_list, gate, _, _, _, _ = self.forward(mel=mel,
                                                                    speaker_ids=speaker_ids, text=text,
                                                                    in_lens=in_lens, out_lens=None)
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

    @classmethod
    def list_available_models(cls) -> PretrainedModelInfo:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        #TODO: this will not work until NGC public URL is fed here. Or we could point the URL to a GCS bucket. 
        """
        list_of_models = []
        model = PretrainedModelInfo(
            pretrained_model_name="Flowtron",
            location="https://prod-model-registry-ngc-bucket.s3.us-west-2.amazonaws.com/org/nvidian/team/sae/models/flowtron/versions/1.5.1/files.zip?response-content-disposition=attachment%3B%20filename%3D%22files.zip%22&response-content-type=application%2Fzip&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEDcaCXVzLXdlc3QtMSJIMEYCIQDNX6R58ee2NBnt2iWjd2E7%2FVDbGAEf4DNhXtSnZ0y3PQIhAOzZt9AJ57HBlDgeI9RNHxR10K9IRxdIsOjKxIiIf91nKvoDCHAQAxoMNzg5MzYzMTM1MDI3IgykdLXIJ3U3Qmg3S%2FEq1wNpPu5Qdju2aZVG0a6O9NhQ2XMnTMxVoGHlUzYOa0B2NzB8l3WPaCPegSrR4MJl048VQ2wFXgrNeeCA0c93mFoEMBafFC0DOVwVzca50OLMWSWeILXuDMdeAElAQEiJrSbMY9o5GYhIRoKoCKxtxWknJlM62FvOLQO3HeTt00XBxQrj%2F7WXq7CL4tShwJWK1OKilj2paeUoRn0xukTP356QvY5%2FLqs3flN6TlmellQ3YnDMmOnHCgFn5dOzgO8aJEJBjXkDV0Ns4gbnPq%2FmTBjtfroVSz2ID8A9Kq9dVxb5NEFvKaj%2FOQIeH9P8DWxlthAQKFEb3o5%2FDO6dw%2FpbSmWBlf%2FWO8XC0om5ydpGPBDMXqRsgDtbFd4nKGBAcKV7Ocn51%2Fyq5zRk63n%2Bo9SHR%2BTTlyEobRNqrooGpdBJYOo%2FsIpc2bm0sqZlPExJUAs8EVXTi7di59hPvf0NSzBXY%2Bb21Jr3Q0PhJ8AIAC9PDRxIVPZzNgZyrgs8c%2FBRLOyTKThBFTs30ShH9swXS2jJoKp11HXpMtXqtRLj%2FK2unqUr%2FNwfNnL%2BCYTiFNoBm5KLaMzWJK2rpH%2BMC6s8JpWBPprPmoqy70DbpSCdS865gt%2FyPZ762yL3nEYw2fvtjwY6pAG%2FLMderRTrq3%2Bk9N6UbK7PLC%2BABkKpdqAg%2F7cm8sysHkTA3cEEzLdzi0GzArqK8UGDEskj5sNebFCAC2FuN94x79h50nAYRSrLIsC1foEOTsbdkhh9E0hoXJtJbCSsXu%2FPKrkZcdQ2c7J%2BQSts306e1xGLOuzrc5JmfSQODSZ4WXPco1%2BL9vt0Upw%2FqUlngPE3z2nrxlHbf9aKnCLVoFh6e8a16Q%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20220203T081231Z&X-Amz-SignedHeaders=host&X-Amz-Expires=3599&X-Amz-Credential=ASIA3PSNVSIZT2D7OXMQ%2F20220203%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Signature=3424e0aeee903038fec173134ff8c4082a5d599d8e80923745f919251dc625ac",
            description="This model is trained on LJSpeech sampled at 22050Hz.",
            class_=cls,
            aliases=["flowtron"],
        )
        list_of_models.append(model)
        return list_of_models
    
    def setup_training_data(self, train_data_config: OmegaConf):
        trainset = instantiate(train_data_config.dataset)            
        self._train_dl = torch.utils.data.DataLoader(trainset, collate_fn=trainset._collate_fn_, **train_data_config.dataloader_params)
  
    def setup_validation_data(self, val_data_config: OmegaConf):
        # Get data, data loaders and 1ollate function ready
        valset = instantiate(val_data_config.dataset)
        self._validation_dl = torch.utils.data.DataLoader(valset, collate_fn=valset._collate_fn_, **val_data_config.dataloader_params)
  
    def training_step(self, batch, batch_idx): 
        
        (mel, spk_ids, txt, in_lens, out_lens,
            gate_target, attn_prior) = batch
        mel, spk_ids, txt = mel.cuda(), spk_ids.cuda(), txt.cuda()
        in_lens, out_lens = in_lens.cuda(), out_lens.cuda()
        gate_target = gate_target.cuda()
        attn_prior = attn_prior.cuda() if attn_prior is not None else None

        if self._cfg.flowtronloss.use_ctc_loss and self.iteration >= self._cfg.trainparams.ctc_loss_start_iter:
            self.apply_ctc = True
        
        (z, log_s_list, gate_pred, attn,
            attn_logprob, mean, log_var, prob) = self.forward(
            mel=mel, speaker_ids=spk_ids, text=txt, in_lens=in_lens, out_lens=out_lens, attn_prior=attn_prior)

        loss_nll, loss_gate, loss_ctc = self.criterion(
            z=z, log_s_list=log_s_list, gate_pred=gate_pred, attn_list=attn,
                attn_logprob=attn_logprob, mean=mean, log_var=log_var, prob=prob,
            gate_target=gate_target, in_lengths=in_lens, out_lengths=out_lens, is_validation=False)
        
        loss = loss_nll + loss_gate

        if self.apply_ctc:
            loss += loss_ctc * self.criterion.ctc_loss_weight

        reduced_loss = loss.item()
        reduced_gate_loss = loss_gate.item()
        reduced_nll_loss = loss_nll.item()
        reduced_ctc_loss = loss_ctc.item()
        
        output = {
            'loss': loss,
            'progress_bar': {'training_loss': reduced_loss},
            'log': {
                'loss': reduced_loss, 
                'gate_loss': reduced_gate_loss, 
                'nll_loss': reduced_nll_loss, 
                'ctc_loss': reduced_ctc_loss
            },
        }

        self.iteration += 1

        return output

    def validation_step(self, batch, batch_idx):
        
        (mel, spk_ids, txt, in_lens, out_lens,
            gate_target, attn_prior) = batch
        
        mel, spk_ids, txt = mel.cuda(), spk_ids.cuda(), txt.cuda()
        in_lens, out_lens = in_lens.cuda(), out_lens.cuda()
        gate_target = gate_target.cuda()
        attn_prior = attn_prior.cuda() if attn_prior is not None else None
        
        with torch.no_grad():
            (z, log_s_list, gate_pred, attn, attn_logprob,
                mean, log_var, prob) = self.forward(
                mel=mel, speaker_ids=spk_ids, text=txt, in_lens=in_lens, out_lens=out_lens, attn_prior=attn_prior)

            loss_nll, loss_gate, loss_ctc = self.criterion(
                z=z, log_s_list=log_s_list, gate_pred=gate_pred, attn_list=attn,
                    attn_logprob=attn_logprob, mean=mean, log_var=log_var, prob=prob,
                gate_target=gate_target, in_lengths=in_lens, out_lengths=out_lens, is_validation=True)
            loss = loss_nll + loss_gate

            if self.apply_ctc:
                loss += loss_ctc * self.criterion.ctc_loss_weight

            reduced_loss = loss.item()
            reduced_gate_loss = loss_gate.item()
            reduced_nll_loss = loss_nll.item()
            reduced_ctc_loss = loss_ctc.item()
            
        if self.logger is not None and self.logger.experiment is not None:
            self._tb_logger = self.logger.experiment
            idx = random.randint(0, len(gate_target) - 1)
            for i in range(len(attn)):
                self._tb_logger.add_image(
                    'attention_weights_{}'.format(i),
                    plot_alignment_to_numpy(attn[i][idx].data.cpu().numpy().T),
                    self.iteration,
                    dataformats='HWC'
                )
        
            if gate_pred is not None:
                gate_pred = gate_pred.transpose(0, 1)[:, :, 0]
                self._tb_logger.add_image(
                    "gate",
                    plot_gate_outputs_to_numpy(
                        gate_target[idx].data.cpu().numpy(),
                        torch.sigmoid(gate_pred[idx]).data.cpu().numpy()),
                    self.iteration, dataformats='HWC')

        return {
            "val_loss": loss,
            "val_loss_nll": reduced_nll_loss,
            "val_loss_gate": reduced_gate_loss,
            "val_loss_ctc": reduced_ctc_loss
        }

    def validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', val_loss_mean)
        return {'val_loss': val_loss_mean}

    def configure_optimizers(self):
        print("Initializing %s optimizer" % (self._cfg.optim_algo))
        if self._cfg.optim_algo == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self._cfg.learning_rate,
                                        weight_decay=self._cfg.weight_decay)
        elif self._cfg.optim_algo == 'RAdam':
            optimizer = RAdam(self.parameters(), lr=self._cfg.learning_rate,
                            weight_decay=self._cfg.weight_decay)
        else:
            print("Unrecognized optimizer %s!" % (self._cfg.optim_algo))
            exit(1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = self._cfg.learning_rate
        return optimizer

    # @typecheck(
    #     input_types={"tokens": NeuralType(('B', 'T'), EmbeddedTextType())},
    #     output_types={"spectrogram_pred": NeuralType(('B', 'D', 'T'), MelSpectrogramType())},
    # )
    def generate_spectrogram(self, tokens, **kwargs):
        speaker_vecs = self.speaker_vecs[None]
        residual = torch.cuda.FloatTensor(1, 80, kwargs['n_frames']).normal_() * kwargs['sigma']
        spectrogram_pred, _ = self.infer(residual, speaker_vecs, tokens, gate_threshold=0.5)
        return spectrogram_pred