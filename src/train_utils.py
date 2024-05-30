import matplotlib.pyplot as plt
import numpy as np
import torch as th
from torch.optim import AdamW
import copy
from tqdm import tqdm
import os
import logging

"""

Utilities for training diffusion model for sequence to sequence text generation task.

Authors: 
Group 14
Yun Chu - 1342245
Thet Htut Aung - 940976
Sally Arnold - 992316

Code is adapted from:

@inproceedings{gong2022diffuseq,
  author = {Gong, Shansan and Li, Mukai and Feng, Jiangtao and Wu, Zhiyong and Kong, Lingpeng},
  booktitle = {International Conference on Learning Representations, ICLR},
  title = {{DiffuSeq}: Sequence to Sequence Text Generation with Diffusion Models},
  year = 2023
}

@article{gong2023diffuseqv2,
  title={DiffuSeq-v2: Bridging Discrete and Continuous Text Spaces for Accelerated Seq2Seq Diffusion Models},
  author={Gong, Shansan and Li, Mukai and Feng, Jiangtao and Wu, Zhiyong and Kong, Lingpeng},
  journal={arXiv preprint arXiv:2310.05793},
  year={2023}
}

"""


class CustomLogger:
    """
    A custom logger that writes logs to a specified file and prints them to the console.
    """
    def __init__(self, log_file='logs.log', log_level=logging.DEBUG):
        # Create a custom logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        # Create handlers
        file_handler = logging.FileHandler(log_file)
        stream_handler = logging.StreamHandler()

        # Set the log level for handlers
        file_handler.setLevel(log_level)
        stream_handler.setLevel(log_level)

        # Create a formatter and set it for handlers
        formatter = logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        # Add handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

        # Suppress matplotlib font-related messages
        logging.getLogger('matplotlib').setLevel(logging.WARNING)

    def get_logger(self):
        return self.logger
    

# Save an instance of logger for this file
log = CustomLogger().get_logger()


class UniformSampler():
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.

    Sampler performs unbiased importance sampling, in which the
    objective's mean is unchanged.
    """

    def __init__(self, diffusion):
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self):
        return self._weights

    def sample(self, batch_size, device):
        """
        Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = th.from_numpy(indices_np).long()#.to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = th.from_numpy(weights_np).float()#.to(device)
        return indices, weights

# Helper functions for training loop
def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)

def zero_grad(model_params):
    """
    Zero the gradients for all the parameters in the model.

    :param model_params (list): A list of all the parameters in the model.
    """
    for param in model_params:
        # Taken from https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer.add_param_group
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()

# Note logger for logging kv_mean not implemented for simplicity
def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        # logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            # logger.logkv_mean(f"{key}_q{quartile}", sub_loss)

class TrainLoop():
    def __init__(self, model, diffusion, data, batch_size, lr, ema_rate, weight_decay=0.0, learning_steps=0, eval_data=None, eval_interval=-1, device=th.device):
        self.model = model.to(device)
        self.ddp_model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = batch_size
        self.lr = lr
        self.ema_rate = [ema_rate] if isinstance(ema_rate, float) else [float(x) for x in ema_rate.split(",")]
        self.schedule_sampler = UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.learning_steps = learning_steps
        self.eval_data = eval_data
        self.eval_interval = eval_interval
        self.device = device
        self.step = 0
        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        self.ema_params = [copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))]
        self.train_losses = []
        self.eval_losses = []

    def _log_grad_norm(self):
        sqsum = 0.0
        for p in self.master_params:
            if p.grad is not None:
                sqsum += (p.grad ** 2).sum().item()

    def _anneal_lr(self):
        if not self.learning_steps:
            return
        frac_done = self.step / self.learning_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def optimize_normal(self):
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def run_step(self, batch, cond):
        batch = batch.to(self.device)
        cond = {k: v.to(self.device) for k, v in cond.items()}
        self.forward_backward(batch, cond)
        self.optimize_normal()

    def forward_only(self, batch, cond):
        with th.no_grad():
            zero_grad(self.model_params)
            for i in range(0, batch.shape[0], self.microbatch):
                micro = batch[i: i + self.microbatch].to(self.device)  # Move batch to GPU
                micro_cond = {k: v[i: i + self.microbatch].to(self.device) for k, v in cond.items()}  # Move cond to GPU
                t, weights = self.schedule_sampler.sample(micro.shape[0], self.device)
                weights = weights.to(self.device)  # Ensure weights is on GPU
                losses = self.diffusion.training_losses(self.ddp_model, micro, t, model_kwargs=micro_cond)
                log_loss_dict(self.diffusion, t, {f"eval_{k}": v * weights for k, v in losses.items()})
                self.eval_losses.append(losses['loss'].mean().item())

    def forward_backward(self, batch, cond):
        zero_grad(self.model_params)
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i: i + self.microbatch].to(self.device)  # Move batch to GPU
            micro_cond = {k: v[i: i + self.microbatch].to(self.device) for k, v in cond.items()}  # Move cond to GPU
            t, weights = self.schedule_sampler.sample(micro.shape[0], self.device)
            weights = weights.to(self.device)  # Ensure weights is on GPU
            losses = self.diffusion.training_losses(self.ddp_model, micro, t, model_kwargs=micro_cond)
            loss = (losses["loss"] * weights).mean()
            log_loss_dict(self.diffusion, t, {k: v * weights for k, v in losses.items()})
            loss.backward()
            self.train_losses.append(loss.item())

    def run_loop(self):
        try:
            with tqdm(total=self.learning_steps, desc="Training Progress") as pbar:
                while not self.learning_steps or self.step < self.learning_steps:
                    batch, cond = next(self.data)
                    self.run_step(batch, cond)
                    if self.eval_data is not None and self.step % self.eval_interval == 0:
                        batch_eval, cond_eval = next(self.eval_data)
                        self.forward_only(batch_eval, cond_eval)
                    self.step += 1
                    pbar.update(1)
        except StopIteration:
            log.info("Data loader exhausted. Saving the model...")

        # Save the model after training is completed
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
        th.save(self.model.state_dict(), './checkpoints/trained_model.pth')
        log.info("Model saved successfully.")
        
        # Plotting the training and validation losses
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss')
        if self.eval_losses:
            plt.plot(self.eval_losses, label='Validation Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss over Time')
        plt.savefig('training_validation_loss.png')  # Save the plot as an image
        plt.close()  # Close the plot to free up memory