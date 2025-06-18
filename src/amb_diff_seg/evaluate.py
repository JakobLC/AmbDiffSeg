import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import tqdm

from data import get_data
from diffusion import create_diffusion_from_args
from collections import defaultdict
from models.unet import create_unet_from_args
from utils import (set_random_seed, get_ambiguous_metrics, load_config)


class DiffusionEvaluator():
    def __init__(self, args, model, device):
        """initialize the evaluator
        Args:
            model (torch.nn.Module): diffusion model to evaluate
            dataloader (torch.utils.data.dataloader.DataLoader): dataloader to sample from
            device (torch.device): device to use for evaluation
        """
        self.args = args
        set_random_seed(self.args.sampling.seed)
        self.model = model
        self.device = device
        self.model.eval()

        self.cgd = create_diffusion_from_args(self.args)

        train,vali,test = get_data(random_crop64=args.unet.random_crop64,
                                   data_folder=args.training.data_folder,
                                   training=False,
                                   batch_size=args.training.batch_size,
                                   return_type="dl")
        self.dataloader = {"train": train,"vali": vali,"test": test}[args.sampling.split]


    def sample(self):
        samples = []
        images = []
        preds = []
        gts = []
        metrics = defaultdict(list)
        if self.args.sampling.num_samples <= 0: #all samples in split
            num_samples = len(self.dataloader.dataset)
        else:
            num_samples = self.args.sampling.num_samples

        N = self.args.sampling.num_preds_per_sample

        if self.args.sampling.tqdm_bar:
            bar = tqdm.tqdm(total=num_samples, desc="Sampling")

        samples_count = 0
        
        for image, gt_mask in self.dataloader:
            B, _, H, W = image.shape
            if samples_count >= num_samples:
                break
            elif B>num_samples-samples_count:
                B = num_samples-samples_count
                image, gt_mask = image[:B], gt_mask[:B]
            samples_count += B

            image = image.to(self.device)

            for j in range(N): 
                x_init = torch.randn((B, 1, H, W), device=self.device)
                with torch.no_grad():
                    pred_mask = self.cgd.sample_loop(model=self.model, 
                                            x_init=x_init, 
                                            im=image, 
                                            num_steps=self.args.sampling.num_timesteps,
                                            sampler_type=self.args.sampling.sampler_type, 
                                            clip_x=self.args.sampling.clip_denoised)
                samples.append((pred_mask>0).cpu())
            
            pred_all = torch.cat(samples[-N:], dim=1).permute(0,2,3,1).numpy()

            samples = []
            gt_mask = gt_mask.permute(0,2,3,1).numpy()==1.0

            if self.args.sampling.save_samples_filename:
                images.extend([image[j].cpu() for j in range(B)])
                gts.extend([gt_mask[j] for j in range(B)])
                preds.extend([pred_all[j] for j in range(B)])
            
            for j in range(B):
                metrics_j = get_ambiguous_metrics(pred_all[j], gt_mask[j])
                metrics_pp_j = get_ambiguous_metrics(pred_all[j], gt_mask[j], postprocess=True)
                metrics_j.update({f"{k}_pp": v for k,v in metrics_pp_j.items()})

                for k,v in metrics_j.items():
                    metrics[k].append(v)
            
            if self.args.sampling.tqdm_bar:
                bar.update(B)
            

        mean_metrics = {k: np.mean(v).item() for k,v in metrics.items()}

        if self.args.sampling.save_samples_filename:
            output_dict = {"image_ids": self.dataloader.dataset.items_ids[:num_samples],
                           "images": images,
                           "preds": preds,
                           "gts": gts,
                           "mean_metrics": mean_metrics,
                           "metrics": metrics}
            torch.save(output_dict, self.args.sampling.save_samples_filename)

        return mean_metrics

def evaluate(args):
    # Load model
    model = create_unet_from_args(args["unet"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load(args.sampling.load_ckpt, weights_only=False)["model"])
    
    evaluator = DiffusionEvaluator(args, model, device)
    metrics = evaluator.sample()
    print("\n".join([f"{k:6}: {v:.4f}" for k,v in metrics.items()]))

if __name__ == "__main__":
    evaluate(load_config())