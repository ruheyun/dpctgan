"""DP-TVAE Synthesizer module.

This module provides a differentially private version of the TVAE synthesizer
using a manual, hook-based implementation of DP-SGD for efficient per-sample
gradient computation.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import Linear, Module, Parameter, ReLU, Sequential

from torch.nn.functional import cross_entropy
from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.base import random_state
from ctgan.synthesizers.tvae import TVAE, Decoder, Encoder, _loss_function
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import opacus
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

from dpsdg.data.dp_data_transformer import DPDataTransformer


class TVAEWrapper(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder


class Encoder(Module):
    """Encoder for the TVAESynthesizer.

    Args:
        data_dim (int):
            Dimensions of the data.
        compress_dims (tuple or list of ints):
            Size of each hidden layer.
        embedding_dim (int):
            Size of the output vector.
    """

    def __init__(self, data_dim, compress_dims, embedding_dim):
        super(Encoder, self).__init__()
        dim = data_dim
        seq = []
        for item in list(compress_dims):
            seq += [
                Linear(dim, item),
                ReLU()
            ]
            dim = item

        self.seq = Sequential(*seq)
        self.fc1 = Linear(dim, embedding_dim)
        self.fc2 = Linear(dim, embedding_dim)

    def forward(self, input_):
        """Encode the passed `input_`."""
        feature = self.seq(input_)
        mu = self.fc1(feature)
        logvar = self.fc2(feature)
        std = torch.exp(0.5 * logvar)
        return mu, std, logvar


class Decoder(Module):
    """Decoder for the TVAESynthesizer.

    Args:
        embedding_dim (int):
            Size of the input vector.
        decompress_dims (tuple or list of ints):
            Size of each hidden layer.
        data_dim (int):
            Dimensions of the data.
    """

    def __init__(self, embedding_dim, decompress_dims, data_dim):
        super(Decoder, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(decompress_dims):
            seq += [Linear(dim, item), ReLU()]
            dim = item

        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)
        # self.sigma = Parameter(torch.ones(data_dim) * 0.1)
        self.register_buffer('sigma', torch.ones(data_dim) * 0.1)

    def forward(self, input_):
        """Decode the passed `input_`."""
        return self.seq(input_), self.sigma


def _loss_function(recon_x, x, sigmas, mu, logvar, output_info, factor):
    st = 0
    loss = []
    for column_info in output_info:
        for span_info in column_info:
            if span_info.activation_fn != 'softmax':
                ed = st + span_info.dim
                std = sigmas[st]
                eq = x[:, st] - torch.tanh(recon_x[:, st])
                loss.append((eq ** 2 / 2 / (std ** 2)).sum())
                loss.append(torch.log(std) * x.size()[0])
                st = ed

            else:
                ed = st + span_info.dim
                loss.append(cross_entropy(
                    recon_x[:, st:ed], torch.argmax(x[:, st:ed], dim=-1), reduction='sum'))
                st = ed

    assert st == recon_x.size()[1]
    KLD = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
    return sum(loss) * factor / x.size()[0], KLD / x.size()[0]


class OPTVAE(TVAE):
    """A TVAE synthesizer with a manual, efficient DP-SGD implementation."""

    def __init__(
        self,
        epsilon=1.0,
        delta=1e-5,
        max_grad_norm=1.0,
        use_gradient_penalty=True,
        use_opacus_noise_mul=False,
        **kwargs,
    ):
        """Create a DP-TVAE synthesizer."""
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.use_opacus_noise_mul = use_opacus_noise_mul
        self._device = torch.device('cuda:0')
    @random_state
    def fit_transformer(self, full_data, discrete_columns):
        self.transformer = DPDataTransformer()
        self.transformer.fit(full_data, discrete_columns)

    @random_state
    def sample(self, num_rows):
        return super().sample(num_rows)

    @random_state
    def fit(self, train_data, discrete_columns=()):
        """Fit the TVAE Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
        """
        random_seeds = torch.empty(2, dtype=int).random_()
        loader_gen = torch.Generator()
        loader_gen.manual_seed(random_seeds[0].item())
        

        self.transformer = DataTransformer()
        self.transformer.fit(train_data, discrete_columns)
        train_data = self.transformer.transform(train_data)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(train_data.astype('float32'))),
            batch_size=self.batch_size, shuffle=True, drop_last=False,
            generator=loader_gen
        )

        data_dim = self.transformer.output_dimensions
        encoder = Encoder(data_dim, self.compress_dims, self.embedding_dim).to(self._device)
        self.decoder = Decoder(self.embedding_dim, self.decompress_dims, data_dim).to(self._device)
        
        if self.epsilon is not None:
            try:
                if not ModuleValidator.is_valid(encoder):
                    encoder = ModuleValidator.fix(encoder)
                if not ModuleValidator.is_valid(self.decoder):
                    self.decoder = ModuleValidator.fix(self.decoder)
            except Exception as e:
                print(f"Warning: ModuleValidator failed: {e}. Proceeding with original models.")

        tvae_module = TVAEWrapper(encoder, self.decoder).to(self._device)

        optimizerAE = Adam(
            tvae_module.parameters(),
            weight_decay=self.l2scale)
        
        if self.epsilon is not None:
            self._privacy_engine = PrivacyEngine()
            tvae_module, optimizerAE, loader = self._privacy_engine.make_private_with_epsilon(
                module=tvae_module,
                optimizer=optimizerAE,
                data_loader=loader,
                epochs=self.epochs,
                target_epsilon=self.epsilon,
                target_delta=self.delta,
                max_grad_norm=self.max_grad_norm,
            )
            print(f'DP Enabled: Target Epsilon={self.epsilon}, Delta={self.delta}')
        else:
            print('DP Disabled: Standard Training')

        with tqdm(range(self.epochs), desc='Training', leave=False) as epoch_bar:
            for _ in epoch_bar:
                for data in loader:
                    optimizerAE.zero_grad(set_to_none=True)
                    real = data[0].to(self._device)
                    mu, std, logvar = tvae_module.encoder(real)
                    eps = torch.randn_like(std)
                    emb = eps * std + mu
                    rec, sigmas = tvae_module.decoder(emb)
                    loss_1, loss_2 = _loss_function(
                        rec, real, sigmas, mu, logvar,
                        self.transformer.output_info_list, self.loss_factor
                    )
                    loss = loss_1 + loss_2
                    loss.backward()
                    optimizerAE.step()
                    if self.epsilon is None:
                        self.decoder.sigma.data.clamp_(0.01, 1.0)
                    else:
                        tvae_module.decoder.sigma.data.clamp_(0.01, 1.0)
                postfix = {'Loss': f'{loss.item():.4f}'}
                if self.epsilon is not None:
                    current_epsilon = self._privacy_engine.get_epsilon(self.delta)
                    postfix['Epsilon'] = f'{current_epsilon:.4f}'

                epoch_bar.set_postfix(postfix)
        self.encoder = tvae_module.encoder
        self.decoder = tvae_module.decoder

        if self.epsilon is not None:
            final_epsilon = self._privacy_engine.get_epsilon(self.delta)
            print(f'Training finished. Final Privacy ({final_epsilon:.4f}, {self.delta})-DP')
