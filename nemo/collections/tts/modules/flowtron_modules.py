from collections.abc import Callable

import numpy as np
import torch
from torch.nn import functional as F

from nemo.collections.tts.modules.submodules import ConvNorm, LinearNorm
from nemo.core.classes import NeuralModule, typecheck
from nemo.core.neural_types.elements import (
    EmbeddedTextType,
    LengthsType
)
from nemo.core.neural_types.neural_type import NeuralType

class Encoder(NeuralModule):
    def __init__(
        self, 
        encoder_n_convolutions: int,
        encoder_embedding_dim: int,
        encoder_kernel_size: int,
        norm_fn: Callable, 
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
                norm_fn(encoder_embedding_dim, affine=True)
            )
            convolutions.append(conv_layer)
        self.convolutions = torch.nn.ModuleList(convolutions)

        self.lstm = torch.nn.LSTM(
            encoder_embedding_dim, int(encoder_embedding_dim / 2), 1, batch_first=True, bidirectional=True
        )

    @property
    def input_types(self):
        return {
            "x": NeuralType(('B', 'D', 'T'), EmbeddedTextType()),
            "in_lens": NeuralType(('B'), LengthsType())
        }
    
    @property
    def output_types(self):
        return {
            "encoder_embedding": NeuralType(('B', 'T', 'D'), EmbeddedTextType()),
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
        norm_fn: Callable, 
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
                norm_fn(encoder_embedding_dim, affine=True)
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
        padded_data = nn.utils.rnn.pack_padded_sequence(padded_data, lens)
        hidden_vectors = recurrent_model(padded_data)[0]
        hidden_vectors, _ = nn.utils.rnn.pad_packed_sequence(hidden_vectors)
        # unsort the results at dim=1 and return
        hidden_vectors = hidden_vectors[:, unsort_idx]
        return hidden_vectors


    @property
    def input_types(self):
        return {
            "x": NeuralType(('B', 'D', 'T'), EmbeddedTextType()),
            "lens": NeuralType(('B'), LengthsType())
        }
    
    @property
    def output_types(self):
        return {
            "encoder_embedding": NeuralType(('B', 'T', 'D'), EmbeddedTextType()),
        }
    
    @typecheck()
    def forward(self, x, lens):
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

        x = x.transpose(0, 1)

        x = torch.nn.utils.rnn.pack_padded_sequence(
            x, in_lens.cpu(), batch_first=True)

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

    #TODO: correct the input and output types
    @property
    def input_types(self):
        return {
            "outputs": NeuralType(('B', 'D', 'T'), EmbeddedTextType()),
            "bs": NeuralType(('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "mean": NeuralType(('B', 'T', 'D'), EmbeddedTextType()),
            "log_var": NeuralType(('B', 'T', 'D'), EmbeddedTextType()),
            "prob": NeuralType(('B', 'T', 'D'), EmbeddedTextType()),
        }

    @typecheck()
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

class ARStep(NeuralModule):
    pass

class ARBackStep(NeuralModule):
    pass