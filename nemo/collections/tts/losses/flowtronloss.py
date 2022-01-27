import torch
from torch.nn import functional as F
from transformers import LogitsProcessorList

from nemo.collections.tts.helpers.helpers import get_mask_from_lengths
from nemo.core.classes import Loss

class AttentionCTCLoss(Loss):
    def __init__(self, blank_logprob=-1):
        super(AttentionCTCLoss, self).__init__()
        self.log_softmax = torch.nn.LogSoftmax(dim=3)
        self.blank_logprob = blank_logprob
        self.CTCLoss = torch.nn.CTCLoss(zero_infinity=True)

    def forward(self, attn, in_lens, out_lens, attn_logprob):
        assert attn_logprob is not None
        key_lens = in_lens
        query_lens = out_lens
        attn_logprob_padded = F.pad(input=attn_logprob,
                                    pad=(1, 0, 0, 0, 0, 0, 0, 0),
                                    value=self.blank_logprob)
        cost_total = 0.0
        for bid in range(attn_logprob.shape[0]):
            target_seq = torch.arange(1, key_lens[bid]+1).unsqueeze(0)
            curr_logprob = attn_logprob_padded[bid].permute(1, 0, 2)[
                :query_lens[bid],
                :,
                :key_lens[bid]+1]
            curr_logprob = self.log_softmax(curr_logprob[None])[0]
            ctc_cost = self.CTCLoss(curr_logprob, target_seq,
                                    input_lengths=query_lens[bid:bid+1],
                                    target_lengths=key_lens[bid:bid+1])
            cost_total += ctc_cost
        cost = cost_total/attn_logprob.shape[0]
        return cost


class FlowtronLoss(Loss):
    def __init__(self, sigma=1.0, gm_loss=False, gate_loss=True,
                 use_ctc_loss=False, ctc_loss_weight=0.0,
                 blank_logprob=-1):
        super(FlowtronLoss, self).__init__()
        self.sigma = sigma
        self.gate_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.gm_loss = gm_loss
        self.gate_loss = gate_loss
        self.use_ctc_loss = use_ctc_loss
        self.ctc_loss_weight = ctc_loss_weight
        self.blank_logprob = blank_logprob
        self.attention_loss = AttentionCTCLoss(
            blank_logprob=self.blank_logprob)

    def forward(self, model_output, gate_target,
                in_lengths, out_lengths, is_validation=False):
        z, log_s_list, gate_pred, attn_list, attn_logprob_list, \
            mean, log_var, prob = model_output

        # create mask for outputs computed on padded data
        mask = get_mask_from_lengths(out_lengths).transpose(0, 1)[..., None]
        mask_inverse = ~mask
        mask, mask_inverse = mask.float(), mask_inverse.float()
        n_mel_dims = z.size(2)
        n_elements = mask.sum()
        for i, log_s in enumerate(log_s_list):
            if i == 0:
                log_s_total = torch.sum(log_s * mask)
            else:
                log_s_total = log_s_total + torch.sum(log_s * mask)

        if self.gm_loss:
            mask = mask[..., None]  # T, B, 1, Dummy
            z = z[..., None]  # T, B, Mel, Dummy
            mean = mean[None]  # Dummy, Dummy or B, Mel, Components
            log_var = log_var[None]  # Dummy, Dummy or B, Mel, Components
            prob = prob[None, :, None]  # Dummy, B, Dummy, Components

            _z = -(z - mean)**2 / (2 * torch.exp(log_var))
            _zmax = _z.max(dim=3, keepdim=True)[0]  # T, B, 80, Dummy
            _z = prob * torch.exp(_z - _zmax) / torch.sqrt(torch.exp(log_var))
            _z = _zmax + torch.log(torch.sum(_z, dim=3, keepdim=True))
            nll = -torch.sum(mask * _z)

            loss = nll - log_s_total
            mask = mask[..., 0]
        else:
            z = z * mask
            loss = torch.sum(z*z)/(2*self.sigma*self.sigma) - log_s_total
        loss = loss / (n_elements * n_mel_dims)

        gate_loss = torch.zeros(1, device=z.device)
        if self.gate_loss > 0:
            gate_pred = (gate_pred * mask)
            gate_pred = gate_pred[..., 0].permute(1, 0)
            gate_loss = self.gate_criterion(gate_pred, gate_target)
            gate_loss = gate_loss.permute(1, 0) * mask[:, :, 0]
            gate_loss = gate_loss.sum() / n_elements

        loss_ctc = torch.zeros_like(gate_loss, device=z.device)
        if self.use_ctc_loss:
            for cur_flow_idx, flow_attn in enumerate(attn_list):
                cur_attn_logprob = attn_logprob_list[cur_flow_idx]
                # flip and send log probs for back step
                if cur_flow_idx % 2 != 0:
                    if cur_attn_logprob is not None:
                        for k in range(cur_attn_logprob.size(0)):
                            cur_attn_logprob[k] = cur_attn_logprob[k].roll(
                                -out_lengths[k].item(),
                                dims=0)
                        cur_attn_logprob = torch.flip(cur_attn_logprob, (1, ))
                cur_flow_ctc_loss = self.attention_loss(
                    flow_attn.unsqueeze(1),
                    in_lengths,
                    out_lengths,
                    attn_logprob=cur_attn_logprob.unsqueeze(1))

                # flip the logprob back to be in backward direction
                if cur_flow_idx % 2 != 0:
                    if cur_attn_logprob is not None:
                        cur_attn_logprob = torch.flip(cur_attn_logprob, (1, ))
                        for k in range(cur_attn_logprob.size(0)):
                            cur_attn_logprob[k] = cur_attn_logprob[k].roll(
                                out_lengths[k].item(),
                                dims=0)
                loss_ctc += cur_flow_ctc_loss

            # make CTC loss independent of number of flows by taking mean
            loss_ctc = loss_ctc / float(len(attn_list))
        return loss, gate_loss, loss_ctc
