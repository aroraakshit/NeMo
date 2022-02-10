from collections.abc import Callable
import math

import numpy as np
import torch
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer

from nemo.collections.tts.modules.submodules import ConvNorm, LinearNorm
from nemo.core.classes import NeuralModule, typecheck
from nemo.core.neural_types.elements import (
    EmbeddedTextType,
    LengthsType,
    MaskType,
    MelSpectrogramType,
    ProbsType,
    VoidType,
    LogprobsType,
    LogitsType
)
from nemo.core.neural_types.neural_type import NeuralType

class Encoder(NeuralModule):
    def __init__(
        self, 
        encoder_n_convolutions: int,
        encoder_embedding_dim: int,
        encoder_kernel_size: int,
    ):
        """
        Flowtron Encoder. Three 1-d convolution banks and a bidirectionsl LSTM

        Args:
            encoder_n_convolutions (int): Number of convolution layers.
            encoder_embedding_dim (int): Final output embedding size.
            encoder_kernel_size (int): Lernel of the convolution front-end.
            norm_fn (Callable): Normalization function
        """

        super(Encoder, self).__init__()
        
        convolutions = []
        for _ in range(encoder_n_convolutions):
            conv_layer = torch.nn.Sequential(
                ConvNorm(
                    encoder_embedding_dim,
                    encoder_embedding_dim,
                    kernel_size=encoder_kernel_size,
                    stride=1,
                    padding=int((encoder_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="relu"
                ),
                torch.nn.InstanceNorm1d(encoder_embedding_dim, affine=True)
            )
            convolutions.append(conv_layer)
        self.convolutions = torch.nn.ModuleList(convolutions)

        self.lstm = torch.nn.LSTM(
            encoder_embedding_dim, int(encoder_embedding_dim / 2), 1, batch_first=True, bidirectional=True
        )

    @property
    def input_types(self):
        return {
            "x": NeuralType(('B', 'D', 'T_text'), EmbeddedTextType()),
            "in_lens": NeuralType(('B'), LengthsType())
        }
    
    @property
    def output_types(self):
        return {
            "outputs": NeuralType(('B', 'T_text', 'D'), EmbeddedTextType()),
        }
    
    @typecheck()
    def forward(self, x, in_lens):
        if x.size()[0] > 1:
            x_embedded = []
            for b_ind in range(x.size()[0]):  # TODO: improve speed
                curr_x = x[b_ind:b_ind+1, :, :in_lens[b_ind]].clone()
                for conv in self.convolutions:
                    curr_x = F.dropout(
                        F.relu(conv(curr_x)),
                        0.5,
                        self.training)
                x_embedded.append(curr_x[0].transpose(0, 1))
            x = torch.nn.utils.rnn.pad_sequence(x_embedded, batch_first=True)
        else:
            for conv in self.convolutions:
                x = F.dropout(
                    F.relu(conv(x)),
                    0.5,
                    self.training)
            x = x.transpose(1, 2)
        x = torch.nn.utils.rnn.pack_padded_sequence(
            x, in_lens.cpu(), batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)

        return outputs

    def infer(self, x):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs

class MelEncoder(NeuralModule):
    def __init__(
        self, 
        encoder_n_convolutions: int,
        encoder_embedding_dim: int,
        encoder_kernel_size: int,
    ):
        """
        Flowtron MelEncoder. Three 1-d convolution banks and a bidirectionsl LSTM

        Args:
            encoder_n_convolutions (int): Number of convolution layers.
            encoder_embedding_dim (int): Final output embedding size.
            encoder_kernel_size (int): Lernel of the convolution front-end.
            norm_fn (Callable): Normalization Function
        """

        super(MelEncoder, self).__init__()
        
        convolutions = []
        for _ in range(encoder_n_convolutions):
            conv_layer = torch.nn.Sequential(
                ConvNorm(80 if _ == 0 else 
                    encoder_embedding_dim,
                    encoder_embedding_dim,
                    kernel_size=encoder_kernel_size,
                    stride=1,
                    padding=int((encoder_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="relu"
                ),
                torch.nn.InstanceNorm1d(encoder_embedding_dim, affine=True)
            )
            convolutions.append(conv_layer)
        self.convolutions = torch.nn.ModuleList(convolutions)

        self.lstm = torch.nn.LSTM(
            encoder_embedding_dim, int(encoder_embedding_dim / 2), 1, bidirectional=True
        )

    def run_padded_sequence(
        self, 
        sorted_idx, 
        unsort_idx, 
        lens, 
        padded_data, 
        recurrent_model
    ):
        """
        Sorts input data by previded ordering (and un-ordering)
        and runs the packed data through the recurrent model

        Args:
            sorted_idx (torch.tensor): 1D sorting index
            unsort_idx (torch.tensor): 1D unsorting index (inverse of sorted_idx)
            lens: lengths of input data (sorted in descending order)
            padded_data (torch.tensor): input sequences (padded)
            recurrent_model (nn.Module): recurrent model through which to run the data
        Returns:
            hidden_vectors (torch.tensor): outputs of the RNN, in the original, unsorted, ordering
        """

        # sort the data by decreasing length using provided index
        # we assume batch index is in dim=1
        padded_data = padded_data[:, sorted_idx]

        # lens argument to cpu because of: https://github.com/pytorch/pytorch/issues/43227 
        padded_data = torch.nn.utils.rnn.pack_padded_sequence(padded_data, lens.cpu())
        hidden_vectors = recurrent_model(padded_data)[0]
        hidden_vectors, _ = torch.nn.utils.rnn.pad_packed_sequence(hidden_vectors)
        # unsort the results at dim=1 and return
        hidden_vectors = hidden_vectors[:, unsort_idx]
        return hidden_vectors


    # @property
    # def input_types(self):
    #     return {
    #         "x": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
    #         "lens": NeuralType(('B'), LengthsType())
    #     }
    
    # @property
    # def output_types(self):
    #     return {
    #         "x": NeuralType(('B', 'T_text', 'D'), EmbeddedTextType()),
    #     }
    
    # @typecheck()
    def forward(self, x, lens):
        if x.size()[0] > 1:
            x_embedded = []
            # TODO: Speed this up without sacrificing correctness
            for b_ind in range(x.size()[0]):
                curr_x = x[b_ind:b_ind+1, :, :lens[b_ind]].clone()
                for conv in self.convolutions:
                    curr_x = F.dropout(
                        F.relu(conv(curr_x)),
                        0.5,
                        self.training)
                x_embedded.append(curr_x[0].transpose(0, 1))
            x = torch.nn.utils.rnn.pad_sequence(x_embedded, batch_first=True)
        else:
            for conv in self.convolutions:
                x = F.dropout(F.relu(conv(x)), 0.5, self.training)
            x = x.transpose(1, 2)

        x = x.transpose(0, 1)

        self.lstm.flatten_parameters()
        if lens is not None:
            # collect decreasing length indices
            lens, ids = torch.sort(lens, descending=True)
            original_ids = [0] * lens.size(0)
            for i in range(len(ids)):
                original_ids[ids[i]] = i
            x = self.run_padded_sequence(ids, original_ids, lens, x, self.lstm)
        else:
            x, _ = self.lstm(x)

        # average pooling over time dimension
        x = torch.mean(x, dim=0)
        return x

    def infer(self, x):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs

class GaussianMixture(NeuralModule):
    def __init__(
        self,
        n_hidden: int,
        n_components: int,
        n_mel_channels: int,
        fixed_gaussian: bool,
        mean_scale: float,
    ):
        """
        Gaussian mixture model for Flowtron, when components > 1.
        """
        super(GaussianMixture, self).__init__()

        self.n_mel_channels = n_mel_channels
        self.n_components = n_components
        self.fixed_gaussian = fixed_gaussian
        self.mean_scale = mean_scale

        # TODO: fuse into one dense n_components * 3
        self.prob_layer = LinearNorm(n_hidden, n_components)

        if not fixed_gaussian:
            self.mean_layer = LinearNorm(
                n_hidden, n_mel_channels * n_components)
            self.log_var_layer = LinearNorm(
                n_hidden, n_mel_channels * n_components)
        else:
            mean = self.generate_mean(n_mel_channels, n_components, mean_scale)
            log_var = self.generate_log_var(n_mel_channels, n_components)
            self.register_buffer('mean', mean.float())
            self.register_buffer('log_var', log_var.float())

    def generate_mean(self, n_dimensions, n_components, scale=3):
        means = torch.eye(n_dimensions).float()
        ids = np.random.choice(
            range(n_dimensions), n_components, replace=False)
        means = means[ids] * scale
        means = means.transpose(0, 1)
        means = means[None]
        return means

    def generate_log_var(self, n_dimensions, n_components):
        log_var = torch.zeros(1, n_dimensions, n_components).float()
        return log_var

    def generate_prob(self):
        return torch.ones(1, 1).float()

    # #TODO: correct the input and output types
    # @property
    # def input_types(self):
    #     return {
    #         "outputs": NeuralType(('B', 'D', 'T'), EmbeddedTextType()),
    #         "bs": NeuralType(('B'), LengthsType()),
    #     }

    # @property
    # def output_types(self):
    #     return {
    #         "mean": NeuralType(('B', 'T', 'D'), EmbeddedTextType()),
    #         "log_var": NeuralType(('B', 'T', 'D'), EmbeddedTextType()),
    #         "prob": NeuralType(('B', 'T', 'D'), EmbeddedTextType()),
    #     }

    # @typecheck()
    def forward(self, outputs, bs):
        prob = torch.softmax(self.prob_layer(outputs), dim=1)

        if not self.fixed_gaussian:
            mean = self.mean_layer(outputs).view(
                bs, self.n_mel_channels, self.n_components)
            log_var = self.log_var_layer(outputs).view(
                bs, self.n_mel_channels, self.n_components)
        else:
            mean = self.mean
            log_var = self.log_var

        return mean, log_var, prob

class Attention(NeuralModule):
    def __init__(
            self, 
            n_mel_channels=80, 
            n_speaker_dim=128,
            n_text_channels=512, 
            n_att_channels=128, 
            temperature=1.0
        ):
        super(Attention, self).__init__()
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=2)
        self.query = LinearNorm(n_mel_channels,
                                n_att_channels, bias=False, w_init_gain='tanh')
        self.key = LinearNorm(n_text_channels+n_speaker_dim,
                              n_att_channels, bias=False, w_init_gain='tanh')
        self.value = LinearNorm(n_text_channels+n_speaker_dim,
                                n_att_channels, bias=False,
                                w_init_gain='tanh')
        self.v = LinearNorm(n_att_channels, 1, bias=False, w_init_gain='tanh')
        self.score_mask_value = -float("inf")

    def compute_attention_posterior(self, attn, attn_prior, mask=None,
                                    eps=1e-20):
        attn_prior = torch.log(attn_prior.float() + eps)
        attn = torch.log(attn.float() + eps)
        attn_posterior = attn + attn_prior

        attn_logprob = attn_posterior.clone()

        if mask is not None:
            attn_posterior.data.masked_fill_(
                mask.transpose(1, 2), self.score_mask_value)

        attn_posterior = self.softmax(attn_posterior)
        return attn_posterior, attn_logprob

    def forward(self, queries, keys, values, mask=None, attn=None,
                attn_prior=None):
        """
        returns:
            attention weights batch x mel_seq_len x text_seq_len
            attention_context batch x featdim x mel_seq_len
            sums to 1 over text_seq_len(keys)
        """
        if attn is None:
            keys = self.key(keys).transpose(0, 1)
            values = self.value(values) if hasattr(self, 'value') else values
            values = values.transpose(0, 1)
            queries = self.query(queries).transpose(0, 1)
            attn = self.v(torch.tanh((queries[:, :, None] + keys[:, None])))
            attn = attn[..., 0] / self.temperature
            if mask is not None:
                attn.data.masked_fill_(mask.transpose(1, 2),
                                       self.score_mask_value)
            attn = self.softmax(attn)

            if attn_prior is not None:
                attn, attn_logprob = self.compute_attention_posterior(
                    attn, attn_prior, mask)
            else:
                attn_logprob = torch.log(attn.float() + 1e-8)

        else:
            attn_logprob = None
            values = self.value(values)
            values = values.transpose(0, 1)

        output = torch.bmm(attn, values)
        output = output.transpose(1, 2)
        return output, attn, attn_logprob

class AttentionConditioningLayer(NeuralModule):
    """Adapted from the LocationLayer in
    https://github.com/NVIDIA/tacotron2/blob/master/model.py
    1D Conv model over a concatenation of the previous attention and the
    accumulated attention values """
    def __init__(self, input_dim=2, attention_n_filters=32,
                 attention_kernel_sizes=[5, 3], attention_dim=640):
        super(AttentionConditioningLayer, self).__init__()

        self.location_conv_hidden = ConvNorm(
            input_dim, attention_n_filters,
            kernel_size=attention_kernel_sizes[0], padding=None, bias=True,
            stride=1, dilation=1, w_init_gain='relu')
        self.location_conv_out = ConvNorm(
            attention_n_filters, attention_dim,
            kernel_size=attention_kernel_sizes[1], padding=None, bias=True,
            stride=1, dilation=1, w_init_gain='sigmoid')
        self.conv_layers = torch.nn.Sequential(self.location_conv_hidden,
                                         torch.nn.ReLU(),
                                         self.location_conv_out,
                                         torch.nn.Sigmoid())

    def forward(self, attention_weights_cat):
        return self.conv_layers(attention_weights_cat)

class DenseLayer(NeuralModule):
    def __init__(self, in_dim=1024, sizes=[1024, 1024]):
        super(DenseLayer, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = torch.nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=True)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = torch.tanh(linear(x))
        return x

class ARStep(NeuralModule):
    def __init__(
        self, 
        n_mel_channels, 
        n_speaker_dim, 
        n_text_dim, 
        n_hidden,
        n_attn_channels, 
        n_lstm_layers,
        add_gate, 
        use_cumm_attention
    ):
        super(ARStep, self).__init__()
        self.use_cumm_attention = use_cumm_attention
        self.conv = torch.nn.Conv1d(n_hidden, 2*n_mel_channels, 1)
        self.conv.weight.data = 0.0*self.conv.weight.data
        self.conv.bias.data = 0.0*self.conv.bias.data
        self.lstm = torch.nn.LSTM(n_hidden+n_attn_channels, n_hidden, n_lstm_layers)
        self.attention_lstm = torch.nn.LSTM(n_mel_channels, n_hidden)
        self.attention_layer = Attention(n_hidden, n_speaker_dim,
                                        n_text_dim, n_attn_channels)
        if self.use_cumm_attention:
            self.attn_cond_layer = AttentionConditioningLayer(
                input_dim=2, attention_n_filters=32,
                attention_kernel_sizes=[5, 3],
                attention_dim=n_text_dim + n_speaker_dim)
        self.dense_layer = DenseLayer(in_dim=n_hidden,
                                    sizes=[n_hidden, n_hidden])
        if add_gate:
            self.gate_threshold = 0.5
            self.gate_layer = LinearNorm(
                n_hidden+n_attn_channels, 1, bias=True,
                w_init_gain='sigmoid')

    def run_padded_sequence(self, sorted_idx, unsort_idx, lens, padded_data,
                            recurrent_model):
        """Sorts input data by previded ordering (and un-ordering) and runs the
        packed data through the recurrent model

        Args:
            sorted_idx (torch.tensor): 1D sorting index
            unsort_idx (torch.tensor): 1D unsorting index (inverse of sorted_idx)
            lens: lengths of input data (sorted in descending order)
            padded_data (torch.tensor): input sequences (padded)
            recurrent_model (nn.Module): recurrent model to run data through
        Returns:
            hidden_vectors (torch.tensor): outputs of the RNN, in the original,
            unsorted, ordering
        """

        # sort the data by decreasing length using provided index
        # we assume batch index is in dim=1
        padded_data = padded_data[:, sorted_idx]
        padded_data = torch.nn.utils.rnn.pack_padded_sequence(padded_data, lens.cpu())
        hidden_vectors = recurrent_model(padded_data)[0]
        hidden_vectors, _ = torch.nn.utils.rnn.pad_packed_sequence(hidden_vectors)
        # unsort the results at dim=1 and return
        hidden_vectors = hidden_vectors[:, unsort_idx]
        return hidden_vectors

    def run_cumm_attn_sequence(self, attn_lstm_outputs, text, mask,
                               attn_prior=None):
        seq_len, bsize, text_feat_dim = text.shape
        # strangely, appending to a list is faster than pre-allocation
        attention_context_all = []
        attention_weights_all = []
        attention_logprobs_all = []
        attn_cumm_tensor = text[:, :, 0:1].permute(1, 2, 0)*0
        attention_weights = attn_cumm_tensor*0
        for i in range(attn_lstm_outputs.shape[0]):
            attn_cat = torch.cat((attn_cumm_tensor, attention_weights), 1)
            attn_cond_vector = self.attn_cond_layer(attn_cat).permute(2, 0, 1)
            output = attn_lstm_outputs[i:i+1:, :]
            (attention_context, attention_weights,
                attention_logprobs) = self.attention_layer(
                output, text*attn_cond_vector, text, mask=mask,
                attn_prior=attn_prior)
            attention_context_all += [attention_context]
            attention_weights_all += [attention_weights]
            attention_logprobs_all += [attention_logprobs]
            attn_cumm_tensor = attn_cumm_tensor + attention_weights
        attention_context_all = torch.cat(attention_context_all, 2)
        attention_weights_all = torch.cat(attention_weights_all, 1)
        attention_logprobs_all = torch.cat(attention_logprobs_all, 1)
        return {'attention_context': attention_context_all,
                'attention_weights': attention_weights_all,
                'attention_logprobs': attention_logprobs_all}

    @property
    def input_types(self):
        return {
            "mel": NeuralType(('T', 'B', 'D'), MelSpectrogramType()),
            "text": NeuralType(('T_text', 'B', 'D'), EmbeddedTextType()),
            "mask": NeuralType(('B', 'T', 'D'), MaskType()),
            "out_lens": NeuralType(('B'), LengthsType()),
            "attn_prior": NeuralType(('B', 'T_spec', 'T_text'), ProbsType(), optional=True),
        }
    
    @property
    def output_types(self):
        return {
            "mel": NeuralType(('T', 'B', 'D'), MelSpectrogramType()),
            "log_s": NeuralType(('T', 'B', 'D'), VoidType()),
            "gates": NeuralType(('T', 'B', 'D'), LogitsType()),
            "attention_weights": NeuralType(('B', 'T', 'T_text'), VoidType()),
            "attention_logprobs": NeuralType(('B', 'T', 'T_text'), LogprobsType()),
        }
    
    @typecheck()
    def forward(self, mel, text, mask, out_lens, attn_prior=None):
        dummy = torch.FloatTensor(1, mel.size(1), mel.size(2)).zero_()
        dummy = dummy.type(mel.type())
        # seq_len x batch x dim
        mel0 = torch.cat([dummy, mel[:-1, :, :]], 0)
        if out_lens is not None:
            # collect decreasing length indices
            lens, ids = torch.sort(out_lens, descending=True)
            original_ids = [0] * lens.size(0)
            for i in range(len(ids)):
                original_ids[ids[i]] = i
            # mel_seq_len x batch x hidden_dim
            attention_hidden = self.run_padded_sequence(
                ids, original_ids, lens, mel0, self.attention_lstm)
        else:
            attention_hidden = self.attention_lstm(mel0)[0]
        if hasattr(self, 'use_cumm_attention') and self.use_cumm_attention:
            cumm_attn_output_dict = self.run_cumm_attn_sequence(
                attention_hidden, text, mask)
            attention_context = cumm_attn_output_dict['attention_context']
            attention_weights = cumm_attn_output_dict['attention_weights']
            attention_logprobs = cumm_attn_output_dict['attention_logprobs']
        else:
            (attention_context, attention_weights,
                attention_logprobs) = self.attention_layer(
                attention_hidden, text, text, mask=mask, attn_prior=attn_prior)

        attention_context = attention_context.permute(2, 0, 1)
        decoder_input = torch.cat((attention_hidden, attention_context), -1)

        gates = None
        if hasattr(self, 'gate_layer'):
            # compute gates before packing
            gates = self.gate_layer(decoder_input)

        if out_lens is not None:
            # reorder, run padded sequence and undo reordering
            lstm_hidden = self.run_padded_sequence(
                ids, original_ids, lens, decoder_input, self.lstm)
        else:
            lstm_hidden = self.lstm(decoder_input)[0]

        lstm_hidden = self.dense_layer(lstm_hidden).permute(1, 2, 0)
        decoder_output = self.conv(lstm_hidden).permute(2, 0, 1)

        log_s = decoder_output[:, :, :mel.size(2)]
        b = decoder_output[:, :, mel.size(2):]
        mel = torch.exp(log_s) * mel + b
        return mel, log_s, gates, attention_weights, attention_logprobs

    def infer(self, residual, text, attns, attn_prior=None):
        attn_cond_vector = 1.0
        if hasattr(self, 'use_cumm_attention') and self.use_cumm_attention:
            attn_cumm_tensor = text[:, :, 0:1].permute(1, 2, 0)*0
            attention_weight = attn_cumm_tensor*0
        attention_weights = []
        total_output = []  # seems 10FPS faster than pre-allocation

        output = None
        attn = None
        dummy = torch.cuda.FloatTensor(
            1, residual.size(1), residual.size(2)).zero_()
        for i in range(0, residual.size(0)):
            if i == 0:
                attention_hidden, (h, c) = self.attention_lstm(dummy)
            else:
                attention_hidden, (h, c) = self.attention_lstm(output, (h, c))

            if hasattr(self, 'use_cumm_attention') and self.use_cumm_attention:
                attn_cat = torch.cat((attn_cumm_tensor, attention_weight), 1)
                attn_cond_vector = self.attn_cond_layer(attn_cat).permute(2, 0, 1)

            attn = None if attns is None else attns[i][None, None]
            attn_prior_i = None if attn_prior is None else attn_prior[:, i][None]

            (attention_context, attention_weight,
                attention_logprob) = self.attention_layer(
                attention_hidden, text * attn_cond_vector, text, attn=attn,
                attn_prior=attn_prior_i)

            if hasattr(self, 'use_cumm_attention') and self.use_cumm_attention:
                attn_cumm_tensor = attn_cumm_tensor + attention_weight

            attention_weights.append(attention_weight)
            attention_context = attention_context.permute(2, 0, 1)
            decoder_input = torch.cat((
                attention_hidden, attention_context), -1)
            if i == 0:
                lstm_hidden, (h1, c1) = self.lstm(decoder_input)
            else:
                lstm_hidden, (h1, c1) = self.lstm(decoder_input, (h1, c1))
            lstm_hidden = self.dense_layer(lstm_hidden).permute(1, 2, 0)
            decoder_output = self.conv(lstm_hidden).permute(2, 0, 1)

            log_s = decoder_output[:, :, :decoder_output.size(2)//2]
            b = decoder_output[:, :, decoder_output.size(2)//2:]
            output = (residual[i, :, :] - b)/torch.exp(log_s)
            total_output.append(output)
            if (hasattr(self, 'gate_layer') and
                    torch.sigmoid(self.gate_layer(decoder_input)) > self.gate_threshold):
                print("Hitting gate limit")
                break
        total_output = torch.cat(total_output, 0)
        return total_output, attention_weights

class ARBackStep(NeuralModule):
    def __init__(self, n_mel_channels, n_speaker_dim, n_text_dim,
                 n_hidden, n_attn_channels, n_lstm_layers,
                 add_gate, use_cumm_attention):
        super(ARBackStep, self).__init__()
        self.ar_step = ARStep(n_mel_channels, n_speaker_dim, n_text_dim, n_hidden,
                               n_attn_channels, n_lstm_layers, add_gate,
                               use_cumm_attention)

    @property
    def input_types(self):
        return {
            "mel": NeuralType(('T', 'B', 'D'), MelSpectrogramType()),
            "text": NeuralType(('T_text', 'B', 'D'), EmbeddedTextType()),
            "mask": NeuralType(('B', 'T', 'D'), MaskType()),
            "out_lens": NeuralType(('B'), LengthsType()),
            "attn_prior": NeuralType(('B', 'T_spec', 'T_text'), ProbsType(), optional=True),
        }
    
    @property
    def output_types(self):
        return {
            "mel": NeuralType(('T', 'B', 'D'), MelSpectrogramType()),
            "log_s": NeuralType(('T', 'B', 'D'), VoidType()),
            "gates": NeuralType(('T', 'B', 'D'), LogitsType()),
            "attention_weights": NeuralType(('B', 'T', 'T_text'), VoidType()),
            "attention_logprobs": NeuralType(('B', 'T', 'T_text'), LogprobsType()),
        }
    
    @typecheck()
    def forward(self, mel, text, mask, out_lens, attn_prior=None):
        mel = torch.flip(mel, (0, ))
        if attn_prior is not None:
            attn_prior = torch.flip(attn_prior, (1, ))  # (B, M, T)
        # backwards flow, send padded zeros back to end
        for k in range(mel.size(1)):
            mel[:, k] = mel[:, k].roll(out_lens[k].item(), dims=0)
            if attn_prior is not None:
                attn_prior[k] = attn_prior[k].roll(out_lens[k].item(), dims=0)

        mel, log_s, gates, attn_out, attention_logprobs = self.ar_step(
            mel=mel, text=text, mask=mask, out_lens=out_lens, attn_prior=attn_prior)

        # move padded zeros back to beginning
        for k in range(mel.size(1)):
            mel[:, k] = mel[:, k].roll(-out_lens[k].item(), dims=0)
            if attn_prior is not None:
                attn_prior[k] = attn_prior[k].roll(-out_lens[k].item(), dims=0)

        if attn_prior is not None:
            attn_prior = torch.flip(attn_prior, (1, ))
        return (torch.flip(mel, (0, )), log_s, gates,
                attn_out, attention_logprobs)

    def infer(self, residual, text, attns, attn_prior=None):
        # only need to flip, no need for padding since bs=1
        if attn_prior is not None:
            # (B, M, T)
            attn_prior = torch.flip(attn_prior, (1, ))

        residual, attention_weights = self.ar_step.infer(
            torch.flip(residual, (0, )), text, attns, attn_prior=attn_prior)

        if attn_prior is not None:
            attn_prior = torch.flip(attn_prior, (1, ))

        residual = torch.flip(residual, (0, ))
        return residual, attention_weights


class RAdam(Optimizer):
    """RAdam optimizer"""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        """
        Init

        :param params: parameters to optimize
        :param lr: learning rate
        :param betas: beta
        :param eps: numerical precision
        :param weight_decay: weight decay weight
        """
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for _ in range(10)]
        super().__init__(params, defaults)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError(
                        'RAdam does not support sparse gradients'
                    )

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = (
                        state['exp_avg_sq'].type_as(p_data_fp32)
                    )

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = (
                        N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    )
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = (
                            group['lr'] *
                            math.sqrt(
                                (1 - beta2_t) * (N_sma - 4) /
                                (N_sma_max - 4) * (N_sma - 2) /
                                N_sma * N_sma_max / (N_sma_max - 2)
                            ) / (1 - beta1 ** state['step'])
                        )
                    else:
                        step_size = group['lr'] / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(
                        -group['weight_decay'] * group['lr'], p_data_fp32
                    )

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size, exp_avg)

                p.data.copy_(p_data_fp32)

        return loss
