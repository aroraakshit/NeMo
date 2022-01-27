
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
from nemo.core.classes.common import PretrainedModelInfo
from nemo.collections.tts.modules.flowtron_submodules import ARStep, ARBackStep
from nemo.collections.tts.data.datalayers import FlowtronDataCollate
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
    fp16_run: bool = MISSING
    use_ctc_loss: bool = MISSING
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

        self.fp16_run = bool(self._cfg.fp16_run)
        self.use_ctc_loss = bool(self._cfg.use_ctc_loss)
        torch.manual_seed(self._cfg.seed)
        torch.cuda.manual_seed(self._cfg.seed)
    
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

    @classmethod
    def list_available_models(cls) -> PretrainedModelInfo:
        #TODO
        return None
    
    def setup_training_data(self, train_data_config: OmegaConf):
        trainset = instantiate(train_data_config.dataset)
        collate_fn = FlowtronDataCollate(
            n_frames_per_step=1, use_attn_prior=trainset.use_attn_prior)
        self._train_dl = torch.utils.data.DataLoader(trainset, collate_fn=collate_fn, **train_data_config.dataloader_params)
  
    def setup_validation_data(self, val_data_config: OmegaConf):
        # Get data, data loaders and 1ollate function ready
        valset = instantiate(val_data_config.dataset)
        collate_fn = FlowtronDataCollate(
            n_frames_per_step=1, use_attn_prior=valset.use_attn_prior)
        self._validation_dl = torch.utils.data.DataLoader(valset, collate_fn=collate_fn, **val_data_config.dataloader_params)
  
    def training_step(self, *args, **kwargs):
        

        criterion = FlowtronLoss(sigma, bool(model_config['n_components']),
                                gate_loss, use_ctc_loss, ctc_loss_weight,
                                blank_logprob)
        model = Flowtron(**model_config).cuda()

        if len(finetune_layers):
            for name, param in model.named_parameters():
                if name in finetune_layers:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        print("Initializing %s optimizer" % (optim_algo))
        if optim_algo == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                        weight_decay=weight_decay)
        elif optim_algo == 'RAdam':
            optimizer = RAdam(model.parameters(), lr=learning_rate,
                            weight_decay=weight_decay)
        else:
            print("Unrecognized optimizer %s!" % (optim_algo))
            exit(1)

        # Load checkpoint if one exists
        iteration = 0
        if warmstart_checkpoint_path != "":
            model = warmstart(warmstart_checkpoint_path, model)

        if checkpoint_path != "":
            model, optimizer, iteration = load_checkpoint(checkpoint_path, model,
                                                        optimizer, ignore_layers)
            iteration += 1  # next iteration is iteration + 1

        if n_gpus > 1:
            model = apply_gradient_allreduce(model)
        print(model)
        scaler = amp.GradScaler(enabled=fp16_run)

        train_loader, valset, collate_fn = prepare_dataloaders(
            data_config, n_gpus, batch_size)

        # Get shared output_directory ready
        if rank == 0 and not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
            print("Output directory", output_directory)

        if with_tensorboard and rank == 0:
            tboard_out_path = os.path.join(output_directory, 'logs')
            print("Setting up Tensorboard log in %s" % (tboard_out_path))
            logger = FlowtronLogger(tboard_out_path)

        # force set the learning rate to what is specified
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

        model.train()
        epoch_offset = max(0, int(iteration / len(train_loader)))
        apply_ctc = False

        # ================ MAIN TRAINNIG LOOP! ===================
        for epoch in range(epoch_offset, epochs):
            print("Epoch: {}".format(epoch))
            for batch in train_loader:
                model.zero_grad()
                (mel, spk_ids, txt, in_lens, out_lens,
                    gate_target, attn_prior) = batch
                mel, spk_ids, txt = mel.cuda(), spk_ids.cuda(), txt.cuda()
                in_lens, out_lens = in_lens.cuda(), out_lens.cuda()
                gate_target = gate_target.cuda()
                attn_prior = attn_prior.cuda() if attn_prior is not None else None

                if use_ctc_loss and iteration >= ctc_loss_start_iter:
                    apply_ctc = True
                with amp.autocast(enabled=fp16_run):
                    (z, log_s_list, gate_pred, attn,
                        attn_logprob, mean, log_var, prob) = model(
                        mel, spk_ids, txt, in_lens, out_lens, attn_prior)

                    loss_nll, loss_gate, loss_ctc = criterion(
                        (z, log_s_list, gate_pred, attn,
                            attn_logprob, mean, log_var, prob),
                        gate_target, in_lens, out_lens, is_validation=False)
                    loss = loss_nll + loss_gate

                    if apply_ctc:
                        loss += loss_ctc * criterion.ctc_loss_weight

                if n_gpus > 1:
                    reduced_loss = reduce_tensor(loss.data, n_gpus).item()
                    reduced_gate_loss = reduce_tensor(
                        loss_gate.data,
                        n_gpus).item()
                    reduced_mle_loss = reduce_tensor(
                        loss_nll.data,
                        n_gpus).item()
                    reduced_ctc_loss = reduce_tensor(
                        loss_ctc.data,
                        n_gpus).item()
                else:
                    reduced_loss = loss.item()
                    reduced_gate_loss = loss_gate.item()
                    reduced_mle_loss = loss_nll.item()
                    reduced_ctc_loss = loss_ctc.item()

                scaler.scale(loss).backward()
                if grad_clip_val > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        grad_clip_val)

                scaler.step(optimizer)
                scaler.update()

                if rank == 0:
                    print("{}:\t{:.9f}".format(
                        iteration,
                        reduced_loss),
                        flush=True)

                if with_tensorboard and rank == 0:
                    logger.add_scalar('training/loss', reduced_loss, iteration)
                    logger.add_scalar(
                        'training/loss_gate',
                        reduced_gate_loss,
                        iteration)
                    logger.add_scalar(
                        'training/loss_nll',
                        reduced_mle_loss,
                        iteration)
                    logger.add_scalar(
                        'training/loss_ctc',
                        reduced_ctc_loss,
                        iteration)
                    logger.add_scalar(
                        'learning_rate',
                        learning_rate,
                        iteration)

                if iteration % iters_per_checkpoint == 0:
                    (val_loss, val_loss_nll, val_loss_gate, val_loss_ctc,
                        attns, gate_pred, gate_target) = \
                        compute_validation_loss(model, criterion, valset,
                                                batch_size, n_gpus, apply_ctc)
                    if rank == 0:
                        print("Validation loss {}: {:9f}  ".format(
                            iteration, val_loss))
                        if with_tensorboard:
                            logger.log_validation(
                                val_loss, val_loss_nll,
                                val_loss_gate, val_loss_ctc,
                                attns, gate_pred, gate_target, iteration)

                        checkpoint_path = "{}/model_{}".format(
                            output_directory, iteration)
                        save_checkpoint(model, optimizer, learning_rate, iteration,
                                        checkpoint_path)

                iteration += 1
        return self.step_('train', *args, **kwargs)

    def validation_step(self, *args, **kwargs):
        return self.step_('val', *args, **kwargs)

     # This is useful for multiple validation data loader setup
    def validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': val_loss_mean}

    def configure_optimizers(self):
        pass
    