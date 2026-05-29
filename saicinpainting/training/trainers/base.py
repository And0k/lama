import copy
import logging
from typing import Dict, Tuple

import pandas as pd
import pytorch_lightning as ptl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DistributedSampler

from saicinpainting.evaluation import make_evaluator
from saicinpainting.training.data.datasets import make_default_train_dataloader, make_default_val_dataloader
from saicinpainting.training.losses.adversarial import make_discrim_loss
from saicinpainting.training.losses.perceptual import PerceptualLoss, ResNetPL
from saicinpainting.training.modules import make_generator, make_discriminator
from saicinpainting.training.visualizers import make_visualizer
from saicinpainting.utils import add_prefix_to_keys, average_dicts, set_requires_grad, flatten_dict, \
    get_has_ddp_rank

LOGGER = logging.getLogger(__name__)


def make_optimizer(parameters, kind='adamw', **kwargs):
    if kind == 'adam':
        optimizer_class = torch.optim.Adam
    elif kind == 'adamw':
        optimizer_class = torch.optim.AdamW
    else:
        raise ValueError(f'Unknown optimizer kind {kind}')
    return optimizer_class(parameters, **kwargs)


def update_running_average(result: nn.Module, new_iterate_model: nn.Module, decay=0.999):
    with torch.no_grad():
        res_params = dict(result.named_parameters())
        new_params = dict(new_iterate_model.named_parameters())

        for k in res_params.keys():
            res_params[k].data.mul_(decay).add_(new_params[k].data, alpha=1 - decay)


def make_multiscale_noise(base_tensor, scales=6, scale_mode='bilinear'):
    batch_size, _, height, width = base_tensor.shape
    cur_height, cur_width = height, width
    result = []
    align_corners = False if scale_mode in ('bilinear', 'bicubic') else None
    for _ in range(scales):
        cur_sample = torch.randn(batch_size, 1, cur_height, cur_width, device=base_tensor.device)
        cur_sample_scaled = F.interpolate(cur_sample, size=(height, width), mode=scale_mode, align_corners=align_corners)
        result.append(cur_sample_scaled)
        cur_height //= 2
        cur_width //= 2
    return torch.cat(result, dim=1)


class BaseInpaintingTrainingModule(ptl.LightningModule):
    def __init__(self, config, use_ddp, *args,  predict_only=False, visualize_each_iters=100,
                 average_generator=False, generator_avg_beta=0.999, average_generator_start_step=30000,
                 average_generator_period=10, store_discr_outputs_for_vis=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.automatic_optimization = False
        LOGGER.info('BaseInpaintingTrainingModule init called')

        self.config = config

        self.generator = make_generator(config, **self.config.generator)
        self.use_ddp = use_ddp

        if not get_has_ddp_rank():
            n_params = sum(p.numel() for p in self.generator.parameters())
            LOGGER.info('Generator: %s (%.1fM params)', self.generator.__class__.__name__, n_params / 1e6)

        if not predict_only:
            self.save_hyperparameters(self.config)
            self.discriminator = make_discriminator(**self.config.discriminator)
            self.adversarial_loss = make_discrim_loss(**self.config.losses.adversarial)
            self.visualizer = make_visualizer(**self.config.visualizer)
            self.val_evaluator = make_evaluator(**self.config.evaluator)
            self.test_evaluator = make_evaluator(**self.config.evaluator)

            if not get_has_ddp_rank():
                n_params = sum(p.numel() for p in self.discriminator.parameters())
                LOGGER.info('Discriminator: %s (%.1fM params)', self.discriminator.__class__.__name__, n_params / 1e6)

            extra_val = self.config.data.get('extra_val', ())
            if extra_val:
                self.extra_val_titles = list(extra_val)
                self.extra_evaluators = nn.ModuleDict({k: make_evaluator(**self.config.evaluator)
                                                       for k in extra_val})
            else:
                self.extra_evaluators = {}

            self.average_generator = average_generator
            self.generator_avg_beta = generator_avg_beta
            self.average_generator_start_step = average_generator_start_step
            self.average_generator_period = average_generator_period
            self.generator_average = None
            self.last_generator_averaging_step = -1
            self.store_discr_outputs_for_vis = store_discr_outputs_for_vis

            if self.config.losses.get("l1", {"weight_known": 0})['weight_known'] > 0:
                self.loss_l1 = nn.L1Loss(reduction='none')

            if self.config.losses.get("l2", {"weight_known": 0}).get('weight_known', 0) > 0:
                self.loss_l2 = nn.MSELoss(reduction='none')

            if self.config.losses.get("mse", {"weight": 0})['weight'] > 0:
                self.loss_mse = nn.MSELoss(reduction='none')
            
            if self.config.losses.perceptual.weight > 0:
                self.loss_pl = PerceptualLoss()

            if self.config.losses.get("resnet_pl", {"weight": 0})['weight'] > 0:
                self.loss_resnet_pl = ResNetPL(**self.config.losses.resnet_pl)
            else:
                self.loss_resnet_pl = None

            domain_cfg = self.config.losses.get("domain", {})
            self.domain_losses = {}
            if domain_cfg.get("smoothness", {}).get("weight", 0) > 0:
                from saicinpainting.training.losses.physical import smoothness_loss
                self.domain_losses["smoothness"] = {
                    "fn": smoothness_loss,
                    "weight": domain_cfg.smoothness.weight,
                }
            if domain_cfg.get("bounds", {}).get("weight", 0) > 0:
                from saicinpainting.training.losses.physical import physical_bounds_loss
                self.domain_losses["bounds"] = {
                    "fn": physical_bounds_loss,
                    "weight": domain_cfg.bounds.weight,
                }
            if domain_cfg.get("gradient", {}).get("weight", 0) > 0:
                from saicinpainting.training.losses.physical import gradient_consistency_loss
                self.domain_losses["gradient"] = {
                    "fn": gradient_consistency_loss,
                    "weight": domain_cfg.gradient.weight,
                }

            # Physics-motivated losses from lama_hydro_simple_end2end
            if domain_cfg.get("stability", {}).get("weight", 0) > 0:
                from saicinpainting.training.losses.physical import hydrostatic_stability_loss
                self.domain_losses["stability"] = {
                    "fn": hydrostatic_stability_loss,
                    "weight": domain_cfg.stability.weight,
                    "channel_index": domain_cfg.stability.get("channel_index", 1),
                    "alpha": domain_cfg.stability.get("alpha", 0.25),
                }

            if domain_cfg.get("bbl", {}).get("weight", 0) > 0:
                from saicinpainting.training.losses.physical import bottom_boundary_layer_loss
                self.domain_losses["bbl"] = {
                    "fn": bottom_boundary_layer_loss,
                    "weight": domain_cfg.bbl.weight,
                    "channel_index": domain_cfg.bbl.get("channel_index", 1),
                }

            if domain_cfg.get("observation_mse", {}).get("weight", 0) > 0:
                from saicinpainting.training.losses.physical import observation_weighted_mse
                self.domain_losses["observation_mse"] = {
                    "fn": observation_weighted_mse,
                    "weight": domain_cfg.observation_mse.weight,
                    "weight_known": domain_cfg.observation_mse.get("weight_known", 1.0),
                    "weight_domain": domain_cfg.observation_mse.get("weight_domain", 0.1),
                }

        self.visualize_each_iters = visualize_each_iters
        self._val_outputs = []
        LOGGER.info('BaseInpaintingTrainingModule init done')

    def configure_optimizers(self):
        discriminator_params = list(self.discriminator.parameters())
        return [
            dict(optimizer=make_optimizer(self.generator.parameters(), **self.config.optimizers.generator)),
            dict(optimizer=make_optimizer(discriminator_params, **self.config.optimizers.discriminator)),
        ]

    def train_dataloader(self):
        nc_cfg = self.config.data.get('nc', {}).get('data', None)
        if nc_cfg is not None and 'dataset' in nc_cfg:
            from torch.utils.data import DataLoader
            from hydra.utils import instantiate
            ds = instantiate(nc_cfg.dataset)
            return DataLoader(ds, batch_size=nc_cfg.get('batch_size', 2),
                              shuffle=True, num_workers=nc_cfg.get('num_workers', 0))
        kwargs = dict(self.config.data.train)
        if self.use_ddp:
            kwargs['ddp_kwargs'] = dict(num_replicas=self.trainer.num_nodes * self.trainer.num_processes,
                                        rank=self.trainer.global_rank,
                                        shuffle=True)
        dataloader = make_default_train_dataloader(**self.config.data.train)
        return dataloader

    def val_dataloader(self):
        nc_cfg = self.config.data.get('nc', {}).get('data', None)
        if nc_cfg is not None and 'dataset' in nc_cfg:
            from torch.utils.data import DataLoader
            from hydra.utils import instantiate
            ds = instantiate(nc_cfg.dataset)
            return [DataLoader(ds, batch_size=nc_cfg.get('val_batch_size', 1),
                               shuffle=False, num_workers=nc_cfg.get('num_workers', 0))]
        res = [make_default_val_dataloader(**self.config.data.val)]

        if self.config.data.visual_test is not None:
            res = res + [make_default_val_dataloader(**self.config.data.visual_test)]
        else:
            res = res + res

        extra_val = self.config.data.get('extra_val', ())
        if extra_val:
            res += [make_default_val_dataloader(**extra_val[k]) for k in self.extra_val_titles]

        return res

    def training_step(self, batch, batch_idx):
        self._is_training_step = True
        opt_gen, opt_discr = self.optimizers()

        # Generator step
        set_requires_grad(self.generator, True)
        set_requires_grad(self.discriminator, False)

        # Forward pass (must be after set_requires_grad so generator output has grad)
        batch = self(batch)

        gen_loss, gen_metrics = self.generator_loss(batch)
        self.manual_backward(gen_loss)
        opt_gen.step()
        opt_gen.zero_grad()

        # Discriminator step
        if self.config.losses.adversarial.weight > 0:
            set_requires_grad(self.generator, False)
            set_requires_grad(self.discriminator, True)
            discr_loss, discr_metrics = self.discriminator_loss(batch)
            self.manual_backward(discr_loss)
            opt_discr.step()
            opt_discr.zero_grad()
        else:
            discr_loss = torch.tensor(0.0)
            discr_metrics = {}

        # Visualization
        metrics = {**gen_metrics, **discr_metrics}
        if self.get_ddp_rank() in (None, 0) and batch_idx % self.visualize_each_iters == 0:
            if self.config.losses.adversarial.weight > 0 and self.store_discr_outputs_for_vis:
                with torch.no_grad():
                    self.store_discr_outputs(batch)
            self.visualizer(self.current_epoch, batch_idx, batch, suffix='_train')

        metrics_prefix = 'train_'
        log_info = add_prefix_to_keys(metrics, metrics_prefix)
        self.log_dict(log_info, on_step=True, on_epoch=False)

        # Generator averaging
        if self.average_generator \
                and self.global_step >= self.average_generator_start_step \
                and self.global_step >= self.last_generator_averaging_step + self.average_generator_period:
            if self.generator_average is None:
                self.generator_average = copy.deepcopy(self.generator)
            else:
                update_running_average(self.generator_average, self.generator, decay=self.generator_avg_beta)
            self.last_generator_averaging_step = self.global_step

        return gen_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        extra_val_key = None
        if dataloader_idx == 0:
            mode = 'val'
        elif dataloader_idx == 1:
            mode = 'test'
        else:
            mode = 'extra_val'
            extra_val_key = self.extra_val_titles[dataloader_idx - 2]
        self._is_training_step = False
        result = self._do_step(batch, batch_idx, mode=mode, extra_val_key=extra_val_key)
        self._val_outputs.append(result)
        return result

    def on_validation_epoch_end(self):
        outputs = self._val_outputs
        self._val_outputs = []
        averaged_logs = average_dicts(step_out['log_info'] for step_out in outputs)
        self.log_dict({k: v.mean() for k, v in averaged_logs.items()})

        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)

        # standard validation
        val_evaluator_states = [s['val_evaluator_state'] for s in outputs if 'val_evaluator_state' in s]
        if val_evaluator_states:
            val_evaluator_res = self.val_evaluator.evaluation_end(states=val_evaluator_states)
            val_evaluator_res_df = pd.DataFrame(val_evaluator_res).stack(1).unstack(0)
            val_evaluator_res_df.dropna(axis=1, how='all', inplace=True)
            LOGGER.info(f'Validation metrics after epoch #{self.current_epoch}, '
                        f'total {self.global_step} iterations:\n{val_evaluator_res_df}')
            for k, v in flatten_dict(val_evaluator_res).items():
                self.log(f'val_{k}', v)

        # standard visual test
        test_evaluator_states = [s['test_evaluator_state'] for s in outputs
                                 if 'test_evaluator_state' in s]
        if test_evaluator_states:
            test_evaluator_res = self.test_evaluator.evaluation_end(states=test_evaluator_states)
            test_evaluator_res_df = pd.DataFrame(test_evaluator_res).stack(1).unstack(0)
            test_evaluator_res_df.dropna(axis=1, how='all', inplace=True)
            LOGGER.info(f'Test metrics after epoch #{self.current_epoch}, '
                        f'total {self.global_step} iterations:\n{test_evaluator_res_df}')
            for k, v in flatten_dict(test_evaluator_res).items():
                self.log(f'test_{k}', v)

        # extra validations
        if self.extra_evaluators:
            for cur_eval_title, cur_evaluator in self.extra_evaluators.items():
                cur_state_key = f'extra_val_{cur_eval_title}_evaluator_state'
                cur_states = [s[cur_state_key] for s in outputs if cur_state_key in s]
                if not cur_states:
                    continue
                cur_evaluator_res = cur_evaluator.evaluation_end(states=cur_states)
                cur_evaluator_res_df = pd.DataFrame(cur_evaluator_res).stack(1).unstack(0)
                cur_evaluator_res_df.dropna(axis=1, how='all', inplace=True)
                LOGGER.info(f'Extra val {cur_eval_title} metrics after epoch #{self.current_epoch}, '
                            f'total {self.global_step} iterations:\n{cur_evaluator_res_df}')
                for k, v in flatten_dict(cur_evaluator_res).items():
                    self.log(f'extra_val_{cur_eval_title}_{k}', v)

    def _do_step(self, batch, batch_idx, mode='train', optimizer_idx=None, extra_val_key=None):
        if optimizer_idx == 0:  # step for generator
            set_requires_grad(self.generator, True)
            set_requires_grad(self.discriminator, False)
        elif optimizer_idx == 1:  # step for discriminator
            set_requires_grad(self.generator, False)
            set_requires_grad(self.discriminator, True)

        batch = self(batch)

        total_loss = 0
        metrics = {}

        if optimizer_idx is None or optimizer_idx == 0:  # step for generator
            total_loss, metrics = self.generator_loss(batch)

        elif optimizer_idx is None or optimizer_idx == 1:  # step for discriminator
            if self.config.losses.adversarial.weight > 0:
                total_loss, metrics = self.discriminator_loss(batch)

        if self.get_ddp_rank() in (None, 0) and (batch_idx % self.visualize_each_iters == 0 or mode == 'test'):
            if self.config.losses.adversarial.weight > 0:
                if self.store_discr_outputs_for_vis:
                    with torch.no_grad():
                        self.store_discr_outputs(batch)
            vis_suffix = f'_{mode}'
            if mode == 'extra_val':
                vis_suffix += f'_{extra_val_key}'
            self.visualizer(self.current_epoch, batch_idx, batch, suffix=vis_suffix)

        metrics_prefix = f'{mode}_'
        if mode == 'extra_val':
            metrics_prefix += f'{extra_val_key}_'
        result = dict(loss=total_loss, log_info=add_prefix_to_keys(metrics, metrics_prefix))
        if mode == 'val':
            result['val_evaluator_state'] = self.val_evaluator.process_batch(batch)
        elif mode == 'test':
            result['test_evaluator_state'] = self.test_evaluator.process_batch(batch)
        elif mode == 'extra_val':
            result[f'extra_val_{extra_val_key}_evaluator_state'] = self.extra_evaluators[extra_val_key].process_batch(batch)

        return result

    def get_current_generator(self, no_average=False):
        if not no_average and not self.training and self.average_generator and self.generator_average is not None:
            return self.generator_average
        return self.generator

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Pass data through generator and obtain at leas 'predicted_image' and 'inpainted' keys"""
        raise NotImplementedError()

    def generator_loss(self, batch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError()

    def discriminator_loss(self, batch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError()

    def store_discr_outputs(self, batch):
        out_size = batch['image'].shape[2:]
        discr_real_out, _ = self.discriminator(batch['image'])
        discr_fake_out, _ = self.discriminator(batch['predicted_image'])
        batch['discr_output_real'] = F.interpolate(discr_real_out, size=out_size, mode='nearest')
        batch['discr_output_fake'] = F.interpolate(discr_fake_out, size=out_size, mode='nearest')
        batch['discr_output_diff'] = batch['discr_output_real'] - batch['discr_output_fake']

    def get_ddp_rank(self):
        world_size = getattr(self.trainer, 'world_size', None)
        if world_size is None:
            world_size = getattr(self.trainer, 'num_processes', 1) * getattr(self.trainer, 'num_nodes', 1)
        return self.trainer.global_rank if world_size > 1 else None
