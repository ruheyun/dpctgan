import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.base import random_state
from ctgan.synthesizers.tvae import TVAE, Decoder, Encoder
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.utils.batch_memory_manager import BatchMemoryManager

from dpsdg.data.dp_data_transformer import DPDataTransformer


class DPTVAE(TVAE):
    """A TVAE synthesizer using Opacus DP-SGD."""

    def __init__(
        self,
        epsilon=1.0,
        delta=1e-5,
        max_grad_norm=1.0,
        use_gradient_penalty=True,
        use_opacus_noise_mul=False,
        max_physical_batch_size=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm

        # 保留你的参数接口，但在纯 Opacus 实现中
        # 不再需要自己计算/添加 noise。
        self.use_opacus_noise_mul = use_opacus_noise_mul

        self.max_physical_batch_size = max_physical_batch_size

        self.privacy_engine = None
        self.optimizer = None
        self._private_model = None

    @random_state
    def fit_transformer(self, full_data, discrete_columns):
        self.transformer = DPDataTransformer()
        self.transformer.fit(full_data, discrete_columns)

    @random_state
    def sample(self, num_rows):
        return super().sample(num_rows)

    @random_state
    def fit(self, train_data, discrete_columns=()):
        """Fit TVAE using Opacus DP-SGD."""

        # ============================================================
        # 1. Random generators
        # ============================================================

        random_seeds = torch.empty(2, dtype=int).random_()

        loader_gen = torch.Generator()
        loader_gen.manual_seed(random_seeds[0].item())

        # ============================================================
        # 2. Data transformation
        # ============================================================

        self.transformer = DataTransformer()
        self.transformer.fit(train_data, discrete_columns)

        train_data = self.transformer.transform(train_data)

        train_tensor = torch.from_numpy(
            train_data.astype("float32")
        )

        train_dataset = TensorDataset(train_tensor)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            generator=loader_gen,
        )

        data_dim = self.transformer.output_dimensions

        # ============================================================
        # 3. Build Encoder + Decoder
        # ============================================================

        encoder = Encoder(
            data_dim,
            self.compress_dims,
            self.embedding_dim,
        ).to(self._device)

        decoder = Decoder(
            self.embedding_dim,
            self.decompress_dims,
            data_dim,
        ).to(self._device)

        # ============================================================
        # 4. Put encoder and decoder into ONE module
        # ============================================================

        class TVAEModel(nn.Module):
            def __init__(self, encoder, decoder):
                super().__init__()
                self.encoder = encoder
                self.decoder = decoder

            def forward(self, real):
                mu, std, logvar = self.encoder(real)

                eps = torch.randn_like(std)
                emb = eps * std + mu

                rec, sigmas = self.decoder(emb)

                return rec, sigmas, mu, std, logvar

        model = TVAEModel(
            encoder=encoder,
            decoder=decoder,
        ).to(self._device)

        # ============================================================
        # 5. Check Opacus compatibility
        # ============================================================

        if not ModuleValidator.is_valid(model):
            errors = ModuleValidator.validate(
                model,
                strict=False,
            )

            print("Opacus validation errors:")
            for error in errors:
                print(error)

            model = ModuleValidator.fix(model)

        ModuleValidator.validate(
            model,
            strict=True,
        )

        # ============================================================
        # 6. Normal optimizer
        # ============================================================

        optimizerAE = Adam(
            model.parameters(),
            weight_decay=self.l2scale,
        )

        # ============================================================
        # 7. PrivacyEngine
        # ============================================================

        self.privacy_engine = PrivacyEngine()

        # Opacus 根据目标 epsilon/delta/epochs/sample_rate
        # 自动计算 noise_multiplier。
        model, optimizerAE, train_loader = (
            self.privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizerAE,
                data_loader=train_loader,
                target_epsilon=self.epsilon,
                target_delta=self.delta,
                epochs=self.epochs,
                max_grad_norm=self.max_grad_norm,
                loss_reduction="mean",
                poisson_sampling=True,
            )
        )

        print(
            f"DP configuration: "
            f"epsilon={self.epsilon}, "
            f"delta={self.delta}, "
            f"max_grad_norm={self.max_grad_norm}"
        )

        print(
            f"noise_multiplier={optimizerAE.noise_multiplier}"
        )

        # 保存 private model
        self._private_model = model

        # TVAE.sample() 后面需要 self.decoder
        # GradSampleModule 内部的 decoder 就是原来的 decoder。
        self.decoder = model._module.decoder

        # 保存 optimizer
        self.optimizer = optimizerAE

        # ============================================================
        # 8. Training log
        # ============================================================

        self.loss_values = pd.DataFrame(
            columns=["Epoch", "Batch", "Loss"]
        )

        iterator = tqdm(
            range(self.epochs),
            disable=(not self.verbose),
        )

        if self.verbose:
            iterator_description = "Loss: {loss:.3f}"
            iterator.set_description(
                iterator_description.format(loss=0)
            )

        # ============================================================
        # 9. Training
        # ============================================================

        for epoch in iterator:

            loss_values = []
            batch_ids = []

            # --------------------------------------------------------
            # Optional BatchMemoryManager
            # --------------------------------------------------------

            if self.max_physical_batch_size is not None:

                data_loader_context = BatchMemoryManager(
                    data_loader=train_loader,
                    max_physical_batch_size=self.max_physical_batch_size,
                    optimizer=optimizerAE,
                )

            else:
                data_loader_context = None

            if data_loader_context is not None:
                data_loader = data_loader_context.__enter__()
            else:
                data_loader = train_loader

            try:

                for id_, data in enumerate(data_loader):

                    real = data[0].to(self._device)

                    # DPDataLoader 可能产生 empty batch
                    if real.shape[0] == 0:
                        continue

                    batch_size = real.shape[0]

                    # ------------------------------------------------
                    # IMPORTANT:
                    # zero_grad BEFORE forward
                    # ------------------------------------------------

                    optimizerAE.zero_grad()

                    # ------------------------------------------------
                    # Forward
                    # ------------------------------------------------

                    rec, sigmas, mu, std, logvar = model(real)

                    # ------------------------------------------------
                    # Reconstruction loss
                    # ------------------------------------------------

                    loss_cols = []
                    loss_sigmas = 0.0

                    st = 0

                    for column_info in self.transformer.output_info_list:

                        for span_info in column_info:

                            ed = st + span_info.dim

                            if span_info.activation_fn != "softmax":

                                sigma = sigmas[st]

                                eq = (
                                    real[:, st]
                                    - torch.tanh(rec[:, st])
                                )

                                loss_cols.append(
                                    eq**2 / (2 * sigma**2)
                                )

                                loss_sigmas = (
                                    loss_sigmas
                                    + torch.log(sigma)
                                )

                            else:

                                loss_cols.append(
                                    cross_entropy(
                                        rec[:, st:ed],
                                        real[:, st:ed].argmax(dim=-1),
                                        reduction="none",
                                    )
                                )

                            st = ed

                    assert st == rec.size()[1]

                    # [batch]
                    loss_1 = (
                        loss_sigmas
                        + torch.stack(loss_cols).sum(dim=0)
                    ) * self.loss_factor

                    # ------------------------------------------------
                    # KL divergence
                    # ------------------------------------------------

                    # [batch]
                    loss_2 = torch.sum(
                        (
                            1
                            + logvar
                            - mu**2
                            - logvar.exp()
                        ) / -2,
                        dim=1,
                    )

                    # ------------------------------------------------
                    # Per-sample loss
                    # ------------------------------------------------

                    # [batch]
                    per_sample_loss = loss_1 + loss_2

                    # =================================================
                    # IMPORTANT:
                    #
                    # 不需要：
                    #   for j in range(batch_size)
                    #   loss.backward()
                    #   clip_grad_norm_()
                    #
                    # Opacus 会自动计算 per-sample gradients、
                    # clipping 和 Gaussian noise。
                    # =================================================

                    loss = per_sample_loss.mean()

                    loss.backward()

                    # Opacus DPOptimizer:
                    #
                    # optimizer.step()
                    #
                    # 内部完成：
                    #   1. per-sample gradient
                    #   2. clipping
                    #   3. gradient aggregation
                    #   4. Gaussian noise
                    #
                    optimizerAE.step()

                    # TVAE 原来的 sigma 限制
                    self.decoder.sigma.data.clamp_(0.01, 1.0)

                    # ------------------------------------------------
                    # Logging
                    # ------------------------------------------------

                    batch_ids.append(id_)

                    loss_values.append(
                        per_sample_loss.detach().mean().item()
                    )

            finally:

                if data_loader_context is not None:
                    data_loader_context.__exit__(
                        None,
                        None,
                        None,
                    )

            # ========================================================
            # Epoch logging
            # ========================================================

            epoch_loss_df = pd.DataFrame(
                {
                    "Epoch": [epoch] * len(batch_ids),
                    "Batch": batch_ids,
                    "Loss": loss_values,
                }
            )

            if not self.loss_values.empty:

                self.loss_values = pd.concat(
                    [
                        self.loss_values,
                        epoch_loss_df,
                    ],
                    ignore_index=True,
                )

            else:

                self.loss_values = epoch_loss_df

            # ========================================================
            # Privacy accounting
            # ========================================================

            epsilon_spent = self.privacy_engine.get_epsilon(
                self.delta
            )

            if self.verbose and loss_values:

                iterator.set_description(
                    iterator_description.format(
                        loss=loss_values[-1]
                    )
                )

                print(
                    f"Epoch {epoch + 1}/{self.epochs} | "
                    f"Loss={loss_values[-1]:.4f} | "
                    f"ε={epsilon_spent:.4f}, "
                    f"δ={self.delta}"
                )