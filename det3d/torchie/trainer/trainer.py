import logging
import os.path as osp
import queue
import sys
import threading
import time
from collections import OrderedDict
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
from det3d import torchie
# from torch.cuda.amp import autocast as autocast

from . import hooks
from .checkpoint import load_checkpoint, save_checkpoint
from .hooks import (
    CheckpointHook,
    Hook,
    IterTimerHook,
    LrUpdaterHook,
    OptimizerHook,
    lr_updater,
)
from .log_buffer import LogBuffer
from .priority import get_priority
from .utils import (
    all_gather,
    get_dist_info,
    get_host_info,
    get_time_str,
    obj_from_dict,
    synchronize,
)

from det3d.core.utils.center_utils import _transpose_and_gather_feat


def fastfocalloss(out, target, ind, mask, cat):
    '''
    Arguments:
      out, target: B x C x H x W
      ind, mask: B x M
      cat (category id for peaks): B x M
    '''
    mask = mask.float()
    gt = torch.pow(1 - target, 4)
    neg_loss = torch.log(1 - out) * torch.pow(out, 2) * gt
    neg_loss = neg_loss.sum()

    pos_pred_pix = _transpose_and_gather_feat(out, ind) # B x M x C
    pos_pred = pos_pred_pix.gather(2, cat.unsqueeze(2)) # B x M
    num_pos = mask.sum()
    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2) * \
               mask.unsqueeze(2)
    pos_loss = pos_loss.sum()
    if num_pos == 0:
      return - neg_loss
    return - (pos_loss + neg_loss) / num_pos

def mse_loss(f_1,f_2):
    feat_1 = f_1.dense()
    N, C, D, H, W = feat_1.shape
    feat_1 = feat_1.view(N, C*D,H,W)
    feat_2 = f_2.dense().view(N, C*D,H,W)
    return F.mse_loss(feat_1,feat_2.detach(),reduction='sum')/(f_2.features.shape[0] * 10)


def distill_reg_loss( output,target, mask, ind):
    pred = _transpose_and_gather_feat(output, ind)
    gt = _transpose_and_gather_feat(target,ind)
    mask = mask.float().unsqueeze(2) 

    loss = F.mse_loss(pred*mask,  gt*mask, reduction='none')
    loss = loss / (mask.sum() + 1e-4)
    loss = loss.transpose(2 ,0).sum(dim=2).sum(dim=1)
    return loss

def example_to_device(example, device, non_blocking=False) -> dict:
    example_torch = {}
    float_names = ["voxels", "bev_map"]
    for k, v in example.items():
        if k in ["anchors", "anchors_mask", "reg_targets", "reg_weights", "labels", "hm",
                "anno_box", "ind", "mask", 'cat']:
            example_torch[k] = [res.to(device, non_blocking=non_blocking) for res in v]
        elif k in [
            "voxels",
            "dense_voxels",
            "bev_map",
            "coordinates",
            "dense_coordinates",
            "num_points",
            "dense_num_points",
            "points",
            "dense_points",
            "num_voxels",
            "dense_num_voxels",
            "cyv_voxels",
            "cyv_num_voxels",
            "cyv_coordinates",
            "cyv_num_points",
            "gt_boxes_and_cls",
            "reconstruction_coordinates",
            "reconstruction_voxels",
            "reconstruction_num_voxels",
            "reconstruction_num_points",
            "reconstruction_coordinates_4",
            "reconstruction_voxels_4",
            "reconstruction_num_voxels_4",
            "reconstruction_num_points_4",
            "reconstruction_coordinates_2",
            "reconstruction_voxels_2",
            "reconstruction_num_voxels_2",
            "reconstruction_num_points_2"         
        ]:
            example_torch[k] = v.to(device, non_blocking=non_blocking)
        elif k == "calib":
            calib = {}
            for k1, v1 in v.items():
                calib[k1] = v1.to(device, non_blocking=non_blocking)
            example_torch[k] = calib
        else:
            example_torch[k] = v

    return example_torch


def parse_second_losses(losses):

    log_vars = OrderedDict()
    loss = sum(losses["loss"])
    for loss_name, loss_value in losses.items():
        if loss_name == "loc_loss_elem" or loss_name == "T_loc_loss_elem":
            log_vars[loss_name] = [[i.item() for i in j] for j in loss_value]
        else:
            log_vars[loss_name] = [i.item() for i in loss_value]

    return loss, log_vars

def parse_second_losses_(T_losses,losses):

    log_vars = OrderedDict()
    loss = sum(losses["loss"]) + sum(T_losses["loss"])
    for loss_name, loss_value in losses.items():
        if loss_name == "loc_loss_elem":
            log_vars[loss_name] = [[i.item() for i in j] for j in loss_value]
        else:
            log_vars[loss_name] = [i.item() for i in loss_value]
    
    for loss_name, loss_value in T_losses.items():
        if loss_name == "loc_loss_elem":
            log_vars['T_'+loss_name] = [[i.item() for i in j] for j in loss_value]
        else:
            log_vars['T_'+loss_name] = [i.item() for i in loss_value]

    return loss, log_vars


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, max_prefetch=1):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(max_prefetch)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class Prefetcher(object):
    def __init__(self, dataloader):
        self.loader = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input = next(self.loader)
        except StopIteration:
            self.next_input = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = example_to_device(
                self.next_input, torch.cuda.current_device(), non_blocking=False
            )

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        self.preload()
        return input


class Trainer(object):
    """ A training helper for PyTorch

    Args:
        model:
        batch_processor:
        optimizer:
        workdir:
        log_level:
        logger:
    """

    def __init__(
        self,
        model,
        batch_processor,
        optimizer=None,
        lr_scheduler=None,
        work_dir=None,
        log_level=logging.INFO,
        logger=None,
        **kwargs,
    ):
        assert callable(batch_processor)
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.batch_processor = batch_processor

        # Create work_dir
        if torchie.is_str(work_dir):
            self.work_dir = osp.abspath(work_dir)
            torchie.mkdir_or_exist(self.work_dir)
        elif work_dir is None:
            self.work_dir = None
        else:
            raise TypeError("'work_dir' must be a str or None")

        # Get model name from the model class
        if hasattr(self.model, "module"):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__

        self._rank, self._world_size = get_dist_info()
        self.timestamp = get_time_str()
        if logger is None:
            self.logger = self.init_logger(work_dir, log_level)
        else:
            self.logger = logger
        self.log_buffer = LogBuffer()

        self.mode = None
        self._hooks = []
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0
        self._max_epochs = 0
        self._max_iters = 0

    @property
    def model_name(self):
        """str: Name of the model, usually the module class name."""
        return self._model_name

    @property
    def rank(self):
        """int: Rank of current process. (distributed training)"""
        return self._rank

    @property
    def world_size(self):
        """int: Number of processes participating in the job.
        (distributed training)"""
        return self._world_size

    @property
    def hooks(self):
        """list[:obj:`Hook`]: A list of registered hooks."""
        return self._hooks

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    @property
    def inner_iter(self):
        """int: Iteration in an epoch."""
        return self._inner_iter

    @property
    def max_epochs(self):
        """int: Maximum training epochs."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Maximum training iterations."""
        return self._max_iters

    def init_optimizer(self, optimizer):
        """Init the optimizer

        Args:
            optimizer (dict or :obj:`~torch.optim.Optimizer`)

        Returns:
            :obj:`~torch.optim.Optimizer`

        Examples:
            >>> optimizer = dict(type='SGD', lr=0.01, momentum=0.9)
            >>> type(runner.init_optimizer(optimizer))
            <class 'torch.optim.sgd.SGD`>
        """
        if isinstance(optimizer, dict):
            optimizer = obj_from_dict(
                optimizer, torch.optim, dict(params=self.model.parameters())
            )
        elif not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(
                "optimizer must be either an Optimizer object or a dict, "
                "but got {}".format(type(optimizer))
            )
        return optimizer

    def _add_file_handler(self, logger, filename=None, mode="w", level=logging.INFO):
        # TODO: move this method out of runner
        file_handler = logging.FileHandler(filename, mode)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
        return logger

    def init_logger(self, log_dir=None, level=logging.INFO):
        """Init the logger.

        Args:

        Returns:
            :obj:`~logging.Logger`: Python logger.
        """
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - % (message)s", level=level
        )
        logger = logging.getLogger(__name__)
        if log_dir and self.rank == 0:
            filename = "{}.log".format(self.timestamp)
            log_file = osp.join(log_dir, filename)
            self._add_file_handler(logger, log_file, level=level)
        return logger

    def current_lr(self):
        if self.optimizer is None:
            raise RuntimeError("lr is not applicable because optimizer does not exist.")
        return [group["lr"] for group in self.optimizer.param_groups]

    def register_hook(self, hook, priority="NORMAL"):
        """Register a hook into the hook list.

        Args:
            hook (:obj:`Hook`)
            priority (int or str or :obj:`Priority`)
        """
        assert isinstance(hook, Hook)
        if hasattr(hook, "priority"):
            raise ValueError('"priority" is a reserved attribute for hooks')
        priority = get_priority(priority)
        hook.priority = priority
        # Insert the hook to a sorted list
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook)

    def build_hook(self, args, hook_type=None):
        if isinstance(args, Hook):
            return args
        elif isinstance(args, dict):
            assert issubclass(hook_type, Hook)
            return hook_type(**args)
        else:
            raise TypeError(
                "'args' must be either a Hook object"
                " or dict, not {}".format(type(args))
            )

    def call_hook(self, fn_name):
        for hook in self._hooks:
            getattr(hook, fn_name)(self)

    def load_checkpoint(self, filename, map_location="cpu", strict=False):
        self.logger.info("load checkpoint from %s", filename)
        return load_checkpoint(self.model, filename, map_location, strict, self.logger)

    def save_checkpoint(
        self, out_dir, filename_tmpl="epoch_{}.pth", save_optimizer=True, meta=None
    ):
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        else:
            meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        linkpath = osp.join(out_dir, "latest.pth")
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # Use relative symlink
        torchie.symlink(filename, linkpath)

    def batch_processor_inline(self, model, data, train_mode, **kwargs):

        if "local_rank" in kwargs:
            device = torch.device(kwargs["local_rank"])
        else:
            device = None

        # data = example_convert_to_torch(data, device=device)
        example = example_to_device(
            data, torch.cuda.current_device(), non_blocking=False
        )

        self.call_hook("after_data_to_device")

        if train_mode:
            losses = model(example, return_loss=True)
            self.call_hook("after_forward")
            loss, log_vars = parse_second_losses(losses)
            del losses

            outputs = dict(
                loss=loss, log_vars=log_vars, num_samples=-1  # TODO: FIX THIS
            )
            self.call_hook("after_parse_loss")

            return outputs
        else:
            return model(example, return_loss=False)

    def train(self, data_loader, epoch, **kwargs):
        
        self.model.train()
        self.mode = "train"
        self.data_loader = data_loader
        self.length = len(data_loader)
        self._max_iters = self._max_epochs * self.length
        self.call_hook("before_train_epoch")

        base_step = epoch * self.length

        # prefetcher = Prefetcher(data_loader)
        # for data_batch in BackgroundGenerator(data_loader, max_prefetch=3):
        for i, data_batch in enumerate(data_loader):
            global_step = base_step + i
            if self.lr_scheduler is not None:

                self.lr_scheduler.step(global_step)

            self._inner_iter = i

            self.call_hook("before_train_iter")


            outputs = self.batch_processor_inline(
                self.model, data_batch, train_mode=True, **kwargs
            )

            if not isinstance(outputs, dict):
                raise TypeError("batch_processor() must return a dict")
            if "log_vars" in outputs:
                self.log_buffer.update(outputs["log_vars"], outputs["num_samples"])
            self.outputs = outputs
            self.call_hook("after_train_iter")
            self._iter += 1

        self.call_hook("after_train_epoch")
        self._epoch += 1

    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = "val"
        self.data_loader = data_loader
        self.call_hook("before_val_epoch")

        self.logger.info(f"work dir: {self.work_dir}")

        if self.rank == 0:
            prog_bar = torchie.ProgressBar(len(data_loader.dataset))

        detections = {}
        cpu_device = torch.device("cpu")

        for i, data_batch in enumerate(data_loader):
            self._inner_iter = i
            self.call_hook("before_val_iter")
            with torch.no_grad():
                outputs = self.batch_processor(
                    self.model, data_batch, train_mode=False, **kwargs
                )
            for output in outputs:
                token = output["metadata"]["token"]
                for k, v in output.items():
                    if k not in [
                        "metadata",
                    ]:
                        output[k] = v.to(cpu_device)
                detections.update(
                    {token: output,}
                )
                if self.rank == 0:
                    for _ in range(self.world_size):
                        prog_bar.update()

        synchronize()

        all_predictions = all_gather(detections)

        if self.rank != 0:
            return

        predictions = {}
        for p in all_predictions:
            predictions.update(p)

        # torch.save(predictions, "final_predictions_debug.pkl")
        # TODO fix evaluation module
        result_dict, _ = self.data_loader.dataset.evaluation(
            predictions, output_dir=self.work_dir
        )

        self.logger.info("\n")
        for k, v in result_dict["results"].items():
            self.logger.info(f"Evaluation {k}: {v}")

        self.call_hook("after_val_epoch")

    def resume(self, checkpoint, resume_optimizer=True, map_location="default"):
        if map_location == "default":
            checkpoint = self.load_checkpoint(
                checkpoint , map_location='cuda:{}'.format(torch.cuda.current_device()) # TODO: FIX THIS!!
            )
        else:
            checkpoint = self.load_checkpoint(checkpoint, map_location=map_location)

        self._epoch = checkpoint["meta"]["epoch"]  
        self._iter = checkpoint["meta"]["iter"]
        if "optimizer" in checkpoint and resume_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.logger.info("resumed epoch %d, iter %d", self.epoch, self.iter)

    def run(self, data_loaders, workflow, max_epochs, **kwargs):
        """ Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`])
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs.
            max_epochs (int)
        """
        assert isinstance(data_loaders, list)
        assert torchie.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)

        self._max_epochs = max_epochs
        work_dir = self.work_dir if self.work_dir is not None else "NONE"
        self.logger.info(
            "Start running, host: %s, work_dir: %s", get_host_info(), work_dir
        )
        self.logger.info("workflow: %s, max: %d epochs", workflow, max_epochs)
        self.call_hook("before_run")

        while self.epoch < max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):
                    if not hasattr(self, mode):
                        raise ValueError(
                            "Trainer has no method named '{}' to run an epoch".format(
                                mode
                            )
                        )
                    epoch_runner = getattr(self, mode)
                elif callable(mode):
                    epoch_runner = mode
                else:
                    raise TypeError(
                        "mode in workflow must be a str or "
                        "callable function not '{}'".format(type(mode))
                    )

                for _ in range(epochs):
                    if mode == "train" and self.epoch >= max_epochs:
                        return
                    elif mode == "val":
                        epoch_runner(data_loaders[i], **kwargs)
                    else:
                        epoch_runner(data_loaders[i], self.epoch, **kwargs)

        # time.sleep(1)
        self.call_hook("after_run")

    def register_lr_hooks(self, lr_config):
        if isinstance(lr_config, LrUpdaterHook):
            self.register_hook(lr_config)
        elif isinstance(lr_config, dict):
            assert "policy" in lr_config
            hook_name = lr_config["policy"].title() + "LrUpdaterHook"
            if not hasattr(lr_updater, hook_name):
                raise ValueError('"{}" does not exist'.format(hook_name))
            hook_cls = getattr(lr_updater, hook_name)
            self.register_hook(hook_cls(**lr_config))
        else:
            raise TypeError(
                "'lr_config' must be eigher a LrUpdaterHook object"
                " or dict, not '{}'".format(type(lr_config))
            )

    def register_logger_hooks(self, log_config):
        log_interval = log_config["interval"]
        for info in log_config["hooks"]:
            logger_hook = obj_from_dict(
                info, hooks, default_args=dict(interval=log_interval)
            )
            self.register_hook(logger_hook, priority="VERY_LOW")

    def register_training_hooks(
        self, lr_config, optimizer_config=None, checkpoint_config=None, log_config=None
    ):
        """Register default hooks for training.

        Default hooks include:
            - LrUpdaterHook
            - OptimizerStepperHook
            - CheckpointSaverHook
            - IterTimerHook
            - LoggerHook(s)
        """
        if optimizer_config is None:
            optimizer_config = {}
        if checkpoint_config is None:
            checkpoint_config = {}
        if lr_config is not None:
            assert self.lr_scheduler is None
            self.register_lr_hooks(lr_config)
        self.register_hook(self.build_hook(optimizer_config, OptimizerHook))
        self.register_hook(self.build_hook(checkpoint_config, CheckpointHook))
        self.register_hook(IterTimerHook())
        if log_config is not None:
            self.register_logger_hooks(log_config)


class TS_Trainer(Trainer):
    def __init__(
        self,
        teacher_model,
        student_model,
        batch_processor,
        optimizer=None,
        lr_scheduler=None,
        work_dir=None,
        log_level=logging.INFO,
        logger=None,
        **kwargs
    ):
        super(TS_Trainer, self).__init__(student_model, batch_processor, optimizer=optimizer, lr_scheduler=lr_scheduler,work_dir=work_dir,log_level=logging.INFO,logger=logger)
        self.T_model = teacher_model
        self.g_step = 0

    
    def load_teacher_checkpoint(self, filename, map_location="cpu", strict=False):
        self.logger.info("load checkpoint from %s", filename)
        return load_checkpoint(self.T_model, filename, map_location, strict, self.logger)

    def save_checkpoint(
        self, out_dir, filename_tmpl="epoch_{}.pth", save_optimizer=True, meta=None
    ):
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        else:
            meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        linkpath = osp.join(out_dir, "latest.pth")
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        torchie.symlink(filename, linkpath)
    
    def save_iter_checkpoint(
        self, out_dir, filename_tmpl="epoch_{}_iter_{}.pth", save_optimizer=True, meta=None
    ):
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self._inner_iter)
        else:
            meta.update(epoch=self.epoch + 1, iter=self._inner_iter)

        filename = filename_tmpl.format(self.epoch + 1, self._inner_iter+1)
        filepath = osp.join(out_dir, filename)
        linkpath = osp.join(out_dir, "latest.pth")
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        torchie.symlink(filename, linkpath)

    def batch_processor_inline(self, T_model, S_model, data, train_mode, global_step, **kwargs):

        if "local_rank" in kwargs:
            device = torch.device(kwargs["local_rank"])
        else:
            device = None

        example = example_to_device(
            data, torch.cuda.current_device(), non_blocking=False
        )

        self.call_hook("after_data_to_device")

        if train_mode:
          
            if T_model.backbone._get_name() == "PointPillarsScatter":
                '''
                Point Pillar
                '''
            
                T_preds, F_D_a, F_D_b = T_model(example, return_loss = False)
                losses, F_S_a, F_S_b, S_preds, mask_loss, offset_loss = S_model(example,return_loss = True)
                # downsample the feature to save memory
                F_S_a =  F.max_pool2d(F_S_a,2,2)
                F_D_a =  F.max_pool2d(F_D_a,2,2)

                inds = F_D_a > 0 
                
                sparse2dense_loss = F.mse_loss(F_S_a[~inds],F_D_a[~inds].detach()) * 10 
                sparse2dense_loss += F.mse_loss((F_S_a)[inds],(F_D_a)[inds].detach()) * 10
                # downsample the feature to save memory
                F_D_b = F.max_pool2d(F_D_b,2,2)
                F_S_b = F.max_pool2d(F_S_b,2,2)
                inds = F_D_b > 0
                sparse2dense_loss += F.mse_loss((F_S_a)[inds],(F_D_a)[inds].detach()) * 10
                sparse2dense_loss += F.mse_loss((F_S_b)[inds],(F_D_b)[inds].detach()) * 10
                sparse2dense_loss += F.mse_loss(F_S_b[~inds],F_D_b[~inds].detach()) * 10 

                KD_hm_loss = fastfocalloss(S_preds[0]['hm'] , F.sigmoid(T_preds[0]['hm']).detach(), example['ind'][0], example['mask'][0], example['cat'][0])
                
                
                Distill_Loss =   sparse2dense_loss + KD_hm_loss 
                losses['loss'][0] += Distill_Loss  +  (mask_loss + offset_loss)* 0.5 
                losses['sparse2dense_loss'] = [sparse2dense_loss.detach().cpu()]
                losses['mask_loss'] = [mask_loss.detach().cpu()]
                losses['reconstruction_loss'] = [offset_loss.detach().cpu()]
                losses['T_hm_loss'] = [fastfocalloss(F.sigmoid(T_preds[0]['hm']).detach() , example['hm'][0] , example['ind'][0], example['mask'][0], example['cat'][0]).detach().cpu()]
                losses['kd_hm_loss'] = [KD_hm_loss.detach().cpu()]
                
            elif T_model.backbone._get_name() == "SpMiddleResNetFHD":
                '''
                CenterPoint
                '''
                
                T_preds, F_D_a, F_D_b = T_model(example, return_loss=False, return_feature=True, return_recon_feature = True)
                losses, F_S_a, F_S_b, S_preds, mask_loss, offset_loss = S_model(example, return_loss=True, return_feature=True)
                
                inds = F_D_a > 0
                sparse2dense_loss = F.mse_loss((F_S_a)[inds],(F_D_a)[inds]) * 10
                sparse2dense_loss += F.mse_loss(F_S_a[~inds],F_D_a[~inds]) * 20 

                inds = F_D_b > 0
                sparse2dense_loss += F.mse_loss((F_S_b)[inds],(F_D_b)[inds]) * 5
                sparse2dense_loss += F.mse_loss(F_S_b[~inds],F_D_b[~inds]) * 20 

               
                KD_hm_loss = fastfocalloss(S_preds[0]['hm'] , F.sigmoid(T_preds[0]['hm']), example['ind'][0], example['mask'][0], example['cat'][0])
                T_preds[0]['anno_box'] = torch.cat((T_preds[0]['reg'], T_preds[0]['height'], T_preds[0]['dim'],
                                                            T_preds[0]['rot']), dim=1)  
                KD_reg_loss = distill_reg_loss(S_preds[0]['anno_box'], T_preds[0]['anno_box'], example['mask'][0],example['ind'][0] )
                if hasattr(S_model, 'module'): # We add this loss after subbmision, and it can be removed to reproduce the results in the paper.
                    KD_reg_loss = (KD_reg_loss*KD_reg_loss.new_tensor(S_model.module.bbox_head.code_weights)).sum() * S_model.module.bbox_head.weight
                else:
                    KD_reg_loss = (KD_reg_loss*KD_reg_loss.new_tensor(S_model.bbox_head.code_weights)).sum() * S_model.bbox_head.weight

                
                
                
                Distill_Loss =    KD_hm_loss + KD_reg_loss  + sparse2dense_loss
                losses['loss'][0] += Distill_Loss +  (mask_loss+offset_loss)
                losses['sparse2dense_loss'] = [sparse2dense_loss.detach().cpu()]
                losses['kd_hm_loss'] = [KD_hm_loss.detach().cpu()]
                losses['kd_reg_loss'] = [KD_reg_loss.detach().cpu()]
                losses['mask_loss'] = [mask_loss.detach().cpu()]
                losses['reconstruction_loss'] = [offset_loss.detach().cpu()]
                losses['T_hm_loss'] = [fastfocalloss(F.sigmoid(T_preds[0]['hm']) , example['hm'][0] , example['ind'][0], example['mask'][0], example['cat'][0]).detach().cpu()]

            else:
                '''
                SECOND
                '''
                T_losses, F_D_a, F_D_b = T_model(example, return_loss=True, return_feature=True, return_recon_feature = True)
                losses, F_S_a, F_S_b, S_preds, mask_loss, offset_loss = S_model(example, return_loss=True, return_feature=True)
                
                inds = F_D_a > 0
                sparse2dense_loss = F.mse_loss((F_S_a)[inds],(F_D_a)[inds].detach()) * 10
                sparse2dense_loss += F.mse_loss(F_S_a[~inds],F_D_a[~inds].detach()) * 20
                inds = F_D_b > 0
                sparse2dense_loss += F.mse_loss((F_S_b)[inds],(F_D_b)[inds].detach()) * 5
                sparse2dense_loss += F.mse_loss(F_S_b[~inds],F_D_b[~inds].detach()) * 20 

                
                Distill_Loss =   sparse2dense_loss 
                losses['loss'][0] +=  Distill_Loss + (mask_loss + offset_loss) * 0.5
                losses['sparse2dense_loss'] = [sparse2dense_loss.detach().cpu()]
                losses['mask_loss'] = [mask_loss.detach().cpu()]
                losses['reconstruction_loss'] = [offset_loss.detach().cpu()]
                losses['T_loc_loss_elem'] = [T_losses['loc_loss_elem'][0]]
                losses['T_dir_loss_reduced'] =[T_losses['dir_loss_reduced'][0]]
                losses['T_loss'] = [T_losses['loss'][0].detach().cpu()]
            self.call_hook("after_forward")
            loss, log_vars = parse_second_losses(losses)
            del losses

            outputs = dict(
                loss=loss, log_vars=log_vars, num_samples=-1  # TODO: FIX THIS
            )
            self.call_hook("after_parse_loss")

            return outputs
        else:
            return S_model(example, return_loss=False)
    

    def train(self, data_loader, epoch, **kwargs):
        self.T_model.eval()
        for param in self.T_model.parameters():
            param.requires_grad = False
        self.model.train()
        self.mode = "train"
        self.data_loader = data_loader
        self.length = len(data_loader)
        self._max_iters = self._max_epochs * self.length
        self.call_hook("before_train_epoch")
        base_step = epoch * self.length
        
        # prefetcher = Prefetcher(data_loader)
        # for data_batch in BackgroundGenerator(data_loader, max_prefetch=3):
        for i, data_batch in enumerate(data_loader):
            global_step = base_step + i
            if self.lr_scheduler is not None:
                #print(global_step)
                self.lr_scheduler.step(global_step)

            self._inner_iter = i

            self.call_hook("before_train_iter") 

            # outputs = self.batch_processor(self.model,
            #                                data_batch,
            #                                train_mode=True,
            #                                **kwargs)
            outputs = self.batch_processor_inline(
                self.T_model, self.model, data_batch, train_mode=True, global_step=base_step+i, **kwargs
            )

            if not isinstance(outputs, dict):
                raise TypeError("batch_processor() must return a dict")
            if "log_vars" in outputs:
                self.log_buffer.update(outputs["log_vars"], outputs["num_samples"])
            self.outputs = outputs
            self.call_hook("after_train_iter")
            self._iter += 1

        self.call_hook("after_train_epoch")
        self._epoch += 1

