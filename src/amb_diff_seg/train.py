import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import tqdm
from pathlib import Path
from torch.optim import AdamW

from data import get_data
from diffusion import create_diffusion_from_args
from models.unet import create_unet_from_args
from models.fp16 import (make_master_params, model_grads_to_master_grads, 
                       master_params_to_model_params, zero_grad,
                       unflatten_master_params)
from utils import (set_random_seed, get_train_metrics, load_config)

INITIAL_LOG_LOSS_SCALE = 20.0

class DiffusionModelTrainer:
    def __init__(self,args):
        self.args = args

        self.seed = set_random_seed(self.args.training.seed)

        train,vali,test = get_data(random_crop64=args.unet.random_crop64,
                            data_folder=args.training.data_folder,
                            training=False,
                            batch_size=args.training.batch_size,
                            return_type="dli")
        self.train_dl = train
        self.vali_dl = vali

        self.cgd = create_diffusion_from_args(self.args)
        self.model = create_unet_from_args(self.args["unet"])

        if torch.cuda.is_available():
            print("CUDA available. Using GPU.")
            self.device = torch.device("cuda")
        else:
            print("WARNING: CUDA not available. Using CPU.")
            self.device = torch.device("cpu")


        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_trainable = f"{n_trainable:,}".replace(",", " ")
        print(f"Number of trainable parameters (UNet): {n_trainable}")

        print("Saving to: "+self.args.training.save_path)

        self.model = self.model.to(self.device)
        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        if self.args.training.use_fp16:
            self.master_params = make_master_params(self.model_params)
            self.model.convert_to_fp16()
    
        self.opt = AdamW(self.master_params, 
                        lr=self.args.training.lr, 
                        weight_decay=self.args.training.weight_decay,
                        betas=self.args.training.betas)

        self.step = 0
        self.log_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.last_vali_iou = float("nan")
        self.logging_metrics = {"train": {},
                                "vali": {}}
        print("Init complete.")
        
    def save_train_ckpt(self):
        assert Path(self.args.training.save_path).parent.exists(), f"Save path: {self.args.training.save_path} does not exist."
        save_dict = {"step":          self.step,
                    "model":          self._master_params_to_state_dict(self.master_params),
                    "seed":           self.seed,
                    "optimizer":      self.opt.state_dict(),
                    "log_loss_scale": self.log_loss_scale,
                    "config":         self.args,
                    "metrics":        self.logging_metrics}
        torch.save(save_dict, self.args.training.save_path)

    def run_train_step(self):
        zero_grad(self.model_params)
        image,mask = next(self.train_dl)
        image = image.to(self.device)
        mask = mask.to(self.device)

        #sample a random mask out of the 4 available ones
        mask = mask[torch.arange(len(mask)),torch.randint(0,4,(len(mask),))][:,None]
        #make mask in the [-1,1] interval
        mask = mask*2-1

        output = self.cgd.train_loss_step(model=self.model,
                                          image=image,
                                          x=mask)
        
        if output["pred_x"].isnan().any():
            print("NaN in output, stopping training.")
            exit()
                
        loss = output["loss"]
        if self.args.training.use_fp16:
            loss_scale = 2 ** self.log_loss_scale
            (loss * loss_scale).backward()
        else:
            loss.backward()
        if self.args.training.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal()
        #logging
        metrics = get_train_metrics(pred=output["pred_x"],gt=mask)
        metrics["loss"] = loss.item()
        return metrics

    def run_vali_steps(self):
        metrics_list = []
        for _ in range(self.args.training.vali_batches):
            image,mask = next(self.vali_dl)
            image = image.to(self.device)
            mask = mask.to(self.device)

            #sample a random mask out of the 4 available ones
            mask = mask[torch.arange(len(mask)),torch.randint(0,4,(len(mask),))][:,None]
            #make mask in the [-1,1] interval
            mask = mask*2-1

            output = self.cgd.train_loss_step(model=self.model,
                                            image=image,
                                            x=mask)
            metrics_list.append(get_train_metrics(pred=output["pred_x"],gt=mask))
        metrics = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0].keys()}
        return metrics
    
    def _update_lr(self):
        lrds = self.args.training.lr_decay_steps
        frac_warmup = np.clip(self.step / (self.args.training.lr_warmup_steps + 1e-14),0.0, 1.0)
        frac_decay = np.clip((self.step - self.args.training.max_iter + lrds) / (lrds + 1e-14), 0.0, 1.0)
        decay_func_dict = {"linear": lambda x: 1-x,
                           "cosine": lambda x: 0.5 * (1.0 + np.cos(np.pi * x))}
        warmup_mult = decay_func_dict["linear"](1-frac_warmup)
        decay_mult = decay_func_dict["cosine"](frac_decay)
        
        lr_new = self.args.training.lr * warmup_mult * decay_mult
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr_new
                
    def optimize_fp16(self):
        if any(is_infinite_and_not_none(p.grad) for p in self.model_params):
            self.log_loss_scale = round(self.log_loss_scale-1,round(-np.log10(self.args.training.fp16_scale_growth)))
            self.last_grad_norm = -1.0
            self.last_clip_ratio = -1.0
            if self.log_loss_scale <= -20:
                print("Loss scale has gotten too small, stopping training.")
                exit()
            return
        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2 ** self.log_loss_scale))
        if self.args.training.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.master_params, self.args.training.clip_grad_norm)
        self._update_lr()
        self.opt.step()
        master_params_to_model_params(self.model_params, self.master_params)
        self.log_loss_scale = round(self.log_loss_scale + self.args.training.fp16_scale_growth,
                                    round(-np.log10(self.args.training.fp16_scale_growth)))

    def optimize_normal(self):
        if self.args.training.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.master_params, self.args.training.clip_grad_norm)
        self._update_lr()
        self.opt.step()

    def train_loop(self):
        self.step += 1
        print("Starting training loop...")
        bar = tqdm.tqdm(total=self.args.training.max_iter, desc="Training", unit="it")
        while self.step <= self.args.training.max_iter:

            self.model.train()
            
            metrics = self.run_train_step()
            self.logging_metrics["train"][self.step] = metrics
            

            if (self.step % self.args.training.vali_interval == 0) and (self.args.training.vali_interval>0):
                vali_metrics = self.run_vali_steps()
                self.last_vali_iou = vali_metrics["iou"]
                self.logging_metrics["vali"][self.step] = vali_metrics

            if (self.step % self.args.training.save_interval == 0) and (self.args.training.save_interval>0):
                self.save_train_ckpt()
            bar.set_postfix_str(f"MSE: {metrics['loss']:.4f}, "
                                f"iou (t/v): {metrics['iou']:.4f}/{self.last_vali_iou:.4f}, "
                                f"LLS: {self.log_loss_scale:.3f}")
            bar.update(1)
            self.step += 1
    
    def _master_params_to_state_dict(self, master_params):
        """converts a list of params (flattened list if fp16) 
        to a state dict based on the model's state dict"""
        if self.args.training.use_fp16:
            master_params = unflatten_master_params(self.model_params, master_params)
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        """converts a state dict to a list of params based on the model's state dict"""
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        if self.args.use_fp16:
            return make_master_params(params)
        else:
            return params
        
def is_infinite_and_not_none(x):
    if x is None:
        return False
    else:
        return torch.isinf(x).any()
    
def train(args):
    trainer = DiffusionModelTrainer(args)
    trainer.train_loop()

if __name__ == "__main__":
    train(load_config())

