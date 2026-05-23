import warnings

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.base import random_state
from ctgan.synthesizers.ctgan import CTGAN, Discriminator, Generator
from torch.utils.data import DataLoader, TensorDataset
import opacus
from opacus import PrivacyEngine
from opacus.utils.stateless_utils import grad_sample_mode  # 核心修复依赖

from dpsdg.data.dp_data_transformer import DPDataTransformer

class DPCTGAN(CTGAN):
    def __init__(
        self,
        log_frequency=False,
        epsilon=0.0,
        delta=1e-5,
        max_grad_norm=1.0,
        use_gradient_penalty=True,
        gp_lambda=10,
        **kwargs
    ):
        """Create a DP-CTGAN synthesizer."""
        super().__init__(
            pac=1,
            log_frequency=log_frequency,
            **kwargs
        )
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.gp_lambda = gp_lambda

    @random_state
    def fit_transformer(self, data, discrete_columns):
        self._validate_discrete_columns(data, discrete_columns)
        self._validate_null_data(data, discrete_columns)

        self._transformer = DPDataTransformer()
        self._transformer.fit(data, discrete_columns)

        transformed_data = self._transformer.transform(data)
        self._data_sampler = DataSampler(
            transformed_data, self._transformer.output_info_list, self._log_frequency
        )

    @random_state
    def sample(self, num_rows):
        """Sample data similar to the training data."""
        return super().sample(n=num_rows)

    @random_state
    def condvec_from_real(self, real_data):
        batch = len(real_data)

        discrete_column_id = np.random.choice(self._data_sampler._n_discrete_columns, batch)
        mask = np.zeros((batch, self._data_sampler._n_discrete_columns), dtype='float32')
        mask[np.arange(batch), discrete_column_id] = 1

        category_id_in_col = real_data[np.arange(batch), discrete_column_id].astype("int")
        category_id = self._data_sampler._discrete_column_cond_st[discrete_column_id] + category_id_in_col

        cond = np.zeros((batch, self._data_sampler._n_categories), dtype='float32')
        cond[np.arange(batch), category_id] = 1

        return cond, mask, discrete_column_id, category_id_in_col

    @random_state
    def fit(self, train_data, discrete_columns=()):
        """Fit the CTGAN Synthesizer models to the training data."""
        random_seeds = torch.empty(2, dtype=int).random_()
        loader_gen = torch.Generator()
        loader_gen.manual_seed(random_seeds[0].item())
        noise_gen = torch.Generator(device=self._device)
        noise_gen.manual_seed(random_seeds[1].item())

        train_data = self._transformer.transform(train_data)
        data_loader = DataLoader(
            TensorDataset(torch.from_numpy(train_data.astype("float32"))),
            batch_size=self._batch_size, shuffle=True, drop_last=False,
            generator=loader_gen
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

        is_dp_enabled = self.epsilon > 0

        if is_dp_enabled:
            privacy_engine = PrivacyEngine()
            
            # 开启 DP 包装
            discriminator, optimizerD, data_loader = privacy_engine.make_private_with_epsilon(
                module=discriminator,
                optimizer=optimizerD,
                data_loader=data_loader,
                target_epsilon=self.epsilon,
                target_delta=self.delta,
                epochs=self._epochs,
                max_grad_norm=self.max_grad_norm,
            )
            print(f"Opacus configured with noise multiplier: {optimizerD.noise_multiplier}")

        self.loss_values = pd.DataFrame(columns=['Epoch', 'Generator Loss', 'Discriminator Loss'])

        epoch_iterator = tqdm(range(self._epochs), disable=(not self._verbose))
        if self._verbose:
            description = 'Gen. ({gen:.2f}) | Discrim. ({dis:.2f})'
            epoch_iterator.set_description(description.format(gen=0, dis=0))

        for i in epoch_iterator:
            for batch_data in data_loader:
                current_batch_size = batch_data[0].shape[0]
                real_data = batch_data[0].to(self._device)

                # ==========================================
                # 1. 训练判别器 (Discriminator)
                # ==========================================
                optimizerD.zero_grad(set_to_none=True)

                c1, m1, col, opt = self.condvec_from_real(real_data.cpu().numpy())
                c1 = torch.from_numpy(c1).to(self._device)
                real_cat = torch.cat([real_data, c1], dim=1)

                mean = torch.zeros(current_batch_size, self._embedding_dim, device=self._device)
                fakez = torch.normal(mean=mean, std=mean+1, generator=noise_gen)
                fakez = torch.cat([fakez, c1], dim=1)
                
                fake = self._generator(fakez).detach()
                fakeact = self._apply_activate(fake)
                fake_cat = torch.cat([fakeact, c1], dim=1)

                y_fake = discriminator(fake_cat)
                y_real = discriminator(real_cat)

                # 标准 WGAN 损失
                loss_d = y_fake.mean() - y_real.mean()

                # 计算梯度惩罚 (Gradient Penalty)
                if self.gp_lambda > 0:
                    alpha = torch.rand(real_cat.size(0), 1, device=self._device)
                    alpha = alpha.repeat(1, real_cat.size(1))
                    interpolates = alpha * real_cat + ((1 - alpha) * fake_cat)
                    interpolates.requires_grad_(True)
                    
                    # 关键修改点：在计算内部 autograd 梯度时，如果启用了 DP，则临时关闭 Opacus 的样本级梯度抓取
                    if is_dp_enabled:
                        with grad_sample_mode(discriminator, enabled=False):
                            disc_interpolates = discriminator(interpolates)
                            interpolate_gradients, = torch.autograd.grad(
                                outputs=disc_interpolates,
                                inputs=interpolates,
                                grad_outputs=torch.ones(disc_interpolates.size(), device=self._device),
                                create_graph=True,
                                retain_graph=True,
                                only_inputs=True,
                            )
                    else:
                        disc_interpolates = discriminator(interpolates)
                        interpolate_gradients, = torch.autograd.grad(
                            outputs=disc_interpolates,
                            inputs=interpolates,
                            grad_outputs=torch.ones(disc_interpolates.size(), device=self._device),
                            create_graph=True,
                            retain_graph=True,
                            only_inputs=True,
                        )

                    grad_penalties = ((interpolate_gradients.norm(2, dim=1) - 1) ** 2).mean() * self.gp_lambda
                    loss_d += grad_penalties

                # 此时安全地触发最终反向传播，Opacus 队列不再为空
                loss_d.backward()
                optimizerD.step()

                # 兼容 Opacus 包装后的参数访问以提取规范梯度范数
                raw_discriminator = discriminator._module if is_dp_enabled else discriminator
                total_norm = torch.cat([p.grad.flatten() for p in raw_discriminator.parameters() if p.grad is not None]).norm(2).item()

                # ==========================================
                # 2. 训练生成器 (Generator)
                # ==========================================
                optimizerG.zero_grad(set_to_none=True)

                c1, m1, col, opt = self._data_sampler.sample_condvec(current_batch_size)
                c1 = torch.from_numpy(c1).to(self._device)
                m1 = torch.from_numpy(m1).to(self._device)

                mean = torch.zeros(current_batch_size, self._embedding_dim, device=self._device)
                fakez = torch.normal(mean=mean, std=mean+1, generator=noise_gen)
                fakez = torch.cat([fakez, c1], dim=1)
                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)
                fake_cat = torch.cat([fakeact, c1], dim=1)
                
                y_fake = discriminator(fake_cat)
                cross_entropy = self._cond_loss(fake, c1, m1)

                loss_g = -torch.mean(y_fake) + cross_entropy

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
                self.loss_values = pd.concat([self.loss_values, epoch_loss_df]).reset_index(drop=True)
            else:
                self.loss_values = epoch_loss_df

            if self._verbose:
                epoch_iterator.set_description(
                    description.format(gen=generator_loss, dis=discriminator_loss)
                )