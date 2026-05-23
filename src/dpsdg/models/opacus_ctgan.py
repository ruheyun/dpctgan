import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

from ctgan.data_sampler import DataSampler
from ctgan.synthesizers.base import random_state
from ctgan.synthesizers.ctgan import (
    CTGAN,
    Discriminator,
    Generator,
)

from opacus import PrivacyEngine

from dpsdg.data.dp_data_transformer import DPDataTransformer


class DPDiscriminator(Discriminator):

    def forward(self, input_):
        assert input_.size()[0] % self.pac == 0

        # REMOVE SIGMOID
        # IMPORTANT for BCEWithLogitsLoss / WGAN stability
        return self.seq(input_.view(-1, self.pacdim))


class OpacusCTGAN(CTGAN):

    def __init__(
        self,
        epsilon=1.0,
        delta=1e-5,
        max_grad_norm=1.0,
        **kwargs,
    ):

        super().__init__(pac=1, **kwargs)

        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm

    @random_state
    def fit_transformer(self, full_data, discrete_columns):

        self._validate_discrete_columns(
            full_data,
            discrete_columns
        )

        self._validate_null_data(
            full_data,
            discrete_columns
        )

        self._transformer = DPDataTransformer()

        self._transformer.fit(
            full_data,
            discrete_columns
        )

    @random_state
    def fit(
        self,
        train_data,
        discrete_columns=(),
        epochs=None
    ):

        self._validate_discrete_columns(
            train_data,
            discrete_columns
        )

        self._validate_null_data(
            train_data,
            discrete_columns
        )

        if epochs is None:
            epochs = self._epochs

        if self._transformer is None:

            warnings.warn(
                "Transformer was not preconstructed--likely privacy leak.",
                UserWarning,
            )

            self.fit_transformer(
                train_data,
                discrete_columns
            )

        transformed_data = self._transformer.transform(train_data)

        dataset = TensorDataset(
            torch.from_numpy(
                transformed_data.astype("float32")
            )
        )

        data_loader = DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=True,
            drop_last=True,
        )

        self._data_sampler = DataSampler(
            transformed_data,
            self._transformer.output_info_list,
            self._log_frequency,
        )

        data_dim = self._transformer.output_dimensions

        self._generator = Generator(
            self._embedding_dim
            + self._data_sampler.dim_cond_vec(),
            self._generator_dim,
            data_dim,
        ).to(self._device)

        discriminator = DPDiscriminator(
            data_dim
            + self._data_sampler.dim_cond_vec(),
            self._discriminator_dim,
            pac=self.pac,
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

        # ==========================================================
        # OPACUS
        # ==========================================================

        privacy_engine = PrivacyEngine()

        discriminator, optimizerD, data_loader = (
            privacy_engine.make_private_with_epsilon(
                module=discriminator,
                optimizer=optimizerD,
                data_loader=data_loader,
                target_epsilon=self.epsilon,
                target_delta=self.delta,
                epochs=epochs,
                max_grad_norm=self.max_grad_norm,
                poisson_sampling=False,
            )
        )

        # ==========================================================

        epoch_iterator = tqdm(
            range(epochs),
            disable=(not self._verbose)
        )

        if self._verbose:

            description = (
                'Gen. ({gen:.2f}) | '
                'Discrim. ({dis:.2f})'
            )

            epoch_iterator.set_description(
                description.format(
                    gen=0,
                    dis=0
                )
            )

        loss_values = []

        for i in epoch_iterator:

            for batch_data in data_loader:

                batch_size = batch_data[0].shape[0]

                real_data = batch_data[0].to(self._device)

                # ==================================================
                # DISCRIMINATOR
                # ==================================================

                optimizerD.zero_grad()

                fakez = torch.randn(
                    batch_size,
                    self._embedding_dim,
                    device=self._device,
                )

                condvec = self._data_sampler.sample_condvec(
                    batch_size
                )

                c1, m1, col, opt = condvec

                c1 = torch.from_numpy(c1).to(self._device)

                fakez = torch.cat(
                    [fakez, c1],
                    dim=1
                )

                fake = self._generator(fakez)

                fakeact = self._apply_activate(fake)

                fake_cat = torch.cat(
                    [fakeact, c1],
                    dim=1
                )

                c2 = c1[
                    np.random.permutation(batch_size)
                ]

                real_cat = torch.cat(
                    [real_data, c2],
                    dim=1
                )

                y_fake = discriminator(fake_cat)
                y_real = discriminator(real_cat)

                # BCEWithLogitsLoss
                loss_real = F.binary_cross_entropy_with_logits(
                    y_real,
                    torch.ones_like(y_real),
                )

                loss_fake = F.binary_cross_entropy_with_logits(
                    y_fake,
                    torch.zeros_like(y_fake),
                )

                loss_d = (
                    loss_real + loss_fake
                ) / 2

                loss_d.backward()

                optimizerD.step()

                # ==================================================
                # GENERATOR
                # ==================================================

                optimizerG.zero_grad()

                fakez = torch.randn(
                    batch_size,
                    self._embedding_dim,
                    device=self._device,
                )

                condvec = self._data_sampler.sample_condvec(
                    batch_size
                )

                c1, m1, col, opt = condvec

                c1 = torch.from_numpy(c1).to(self._device)

                m1 = torch.from_numpy(m1).to(self._device)

                fakez = torch.cat(
                    [fakez, c1],
                    dim=1
                )

                fake = self._generator(fakez)

                fakeact = self._apply_activate(fake)

                y_fake = discriminator(
                    torch.cat(
                        [fakeact, c1],
                        dim=1
                    )
                )

                cross_entropy = self._cond_loss(
                    fake,
                    c1,
                    m1
                )

                loss_g = (
                    F.binary_cross_entropy_with_logits(
                        y_fake,
                        torch.ones_like(y_fake),
                    )
                    + cross_entropy
                )

                loss_g.backward()

                optimizerG.step()

                generator_loss = (
                    loss_g.detach()
                    .cpu()
                    .item()
                )

                discriminator_loss = (
                    loss_d.detach()
                    .cpu()
                    .item()
                )

                loss_values.append([
                    i,
                    generator_loss,
                    discriminator_loss
                ])

                if self._verbose:

                    epoch_iterator.set_description(
                        description.format(
                            gen=generator_loss,
                            dis=discriminator_loss,
                        )
                    )

        self.loss_values = pd.DataFrame(
            loss_values,
            columns=[
                'Epoch',
                'Generator Loss',
                'Discriminator Loss'
            ]
        )

    @random_state
    def sample(self, num_rows):

        return super().sample(n=num_rows)