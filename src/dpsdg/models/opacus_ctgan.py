import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.base import random_state
from ctgan.synthesizers.ctgan import CTGAN, Discriminator, Generator
from opacus import PrivacyEngine
from torch.utils.data import DataLoader, TensorDataset
from dpsdg.data.dp_data_transformer import DPDataTransformer

class OpacusCTGAN(CTGAN):
    def __init__(
        self,
        epsilon=None,
        delta=1e-5,
        max_grad_norm=1.0,
        **kwargs
    ):
        """Create a DP-CTGAN synthesizer."""
        super().__init__(log_frequency=True, pac=1, **kwargs)
        self.max_grad_norm = max_grad_norm
        self.epsilon = epsilon
        self.delta = delta

    @random_state
    def fit_transformer(self, data, discrete_columns):
        self._validate_discrete_columns(data, discrete_columns)
        self._validate_null_data(data, discrete_columns)

        self._transformer = DataTransformer()
        self._transformer.fit(data, discrete_columns)

    @random_state
    def sample(self, num_rows):
        """Sample data similar to the training data."""
        return super().sample(n=num_rows)

    @random_state
    def fit(self, train_data, discrete_columns=()):
        """Fit the CTGAN Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
        """
        random_seeds = torch.empty(2, dtype=int).random_()
        loader_gen = torch.Generator()
        loader_gen.manual_seed(random_seeds[0].item())
        noise_gen = torch.Generator(device=self._device)
        noise_gen.manual_seed(random_seeds[1].item())

        train_data = self._transformer.transform(train_data)
        data_loader = DataLoader(
            TensorDataset(torch.from_numpy(train_data.astype("float32"))),
            batch_size=self._batch_size, shuffle=True, drop_last=True,
            generator=loader_gen
        )

        self._data_sampler = DataSampler(
            train_data, self._transformer.output_info_list, self._log_frequency
        )

        data_dim = self._transformer.output_dimensions

        self._generator = Generator(
            self._embedding_dim + self._data_sampler.dim_cond_vec(), self._generator_dim, data_dim
        ).to(self._device)

        discriminator = Discriminator(
            data_dim + self._data_sampler.dim_cond_vec(), self._discriminator_dim, pac=self.pac
        ).to(self._device)

        optimizerG = torch.optim.Adam(
            self._generator.parameters(),
            lr=self._generator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._generator_decay,
        )

        optimizerD = torch.optim.Adam(
            discriminator.parameters(),
            lr=self._discriminator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._discriminator_decay,
        )

        #_discriminator = discriminator

        privacy_engine = PrivacyEngine()
        privacy_args = {
            'module': discriminator,
            'optimizer': optimizerD,
            'data_loader': data_loader,
            'max_grad_norm': self.max_grad_norm,
            'poisson_sampling': False, # XXX
        }
        if self.epsilon:
            privacy_args['target_epsilon'] = self.epsilon
            privacy_args['target_delta'] = self.delta
            privacy_args['epochs'] = self._epochs
            discriminator, optimizerD, data_loader = privacy_engine.make_private_with_epsilon(**privacy_args)
        else:
            privacy_args['noise_multiplier'] = 0
            discriminator, optimizerD, data_loader = privacy_engine.make_private(**privacy_args)

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        self.loss_values = pd.DataFrame(columns=['Epoch', 'Generator Loss', 'Discriminator Loss'])

        epoch_iterator = tqdm(range(self._epochs), disable=(not self._verbose))
        if self._verbose:
            description = 'Gen. ({gen:.2f}) | Discrim. ({dis:.2f})'
            epoch_iterator.set_description(description.format(gen=0, dis=0))

        for i in epoch_iterator:
            for batch_data in data_loader:
                batch_size = batch_data[0].shape[0]

                # Discriminator
                c1, _, _, _ = self._data_sampler.sample_condvec(batch_size)
                c1 = torch.from_numpy(c1).to(self._device)

                fakez = torch.normal(mean=mean, std=std, generator=noise_gen)
                fakez = torch.cat([fakez, c1], dim=1)
                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)
                fake_cat = torch.cat([fakeact, c1], dim=1)

                real_data = batch_data[0].to(self._device)
                real_cat = torch.cat([real_data, c1], dim=1)

                optimizerD.zero_grad(set_to_none=False)
                y_fake = discriminator(fake_cat)
                y_real = discriminator(real_cat)
                loss_fake = torch.mean(y_fake)
                loss_real = -torch.mean(y_real)
                #pen = _discriminator.calc_gradient_penalty(real_cat, fake_cat, self._device, self.pac)
                loss_d = (loss_fake + loss_real) / 2
                loss_d.backward()

                optimizerD.step()
                total_norm = torch.cat([p.grad.flatten() for p in discriminator.parameters()]).norm(2).item()
                #print(i, loss_fake.item(), loss_real.item(), total_norm)

                # Generator
                fakez = torch.normal(mean=mean, std=std, generator=noise_gen)
                c1, m1, col, opt = self._data_sampler.sample_condvec(self._batch_size)

                c1 = torch.from_numpy(c1).to(self._device)
                m1 = torch.from_numpy(m1).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)
                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)
                fake_cat = torch.cat([fakeact, c1], dim=1)
                y_fake = discriminator(fake_cat)

                cross_entropy = self._cond_loss(fake, c1, m1)

                loss_g = -torch.mean(y_fake) + cross_entropy

                optimizerG.zero_grad(set_to_none=False)
                loss_g.backward()
                optimizerG.step()

            generator_loss = loss_g.detach().cpu().item()
            discriminator_loss = loss_d.detach().cpu().item()

            epoch_loss_df = pd.DataFrame({
                'Epoch': [i],
                'Generator Loss': [generator_loss],
                'Discriminator Loss': [discriminator_loss],
                'Discriminator Grad Norm': [total_norm],
            })
            if not self.loss_values.empty:
                self.loss_values = pd.concat([self.loss_values, epoch_loss_df]).reset_index(
                    drop=True
                )
            else:
                self.loss_values = epoch_loss_df

            if self._verbose:
                epoch_iterator.set_description(
                    description.format(gen=generator_loss, dis=discriminator_loss)
                )