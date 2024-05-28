# pip install denoising_diffusion_pytorch

import torch as th
import numpy as np
from denoising_diffusion_pytorch import GaussianDiffusion as BaseGaussianDiffusion

class CustomGaussianDiffusion(BaseGaussianDiffusion):
    def __init__(self, model, betas):
        """
        Initialize the Custom Gaussian Diffusion model.
        
        :param model: The transformer model used for generating predictions.
        :param betas: The beta schedule for the diffusion process.
        """
        super().__init__(
            model=model,
            image_size=None,  # Not applicable for text
            timesteps=len(betas),
            sampling_timesteps=None
        )
        self.betas = betas
        self.num_timesteps = len(betas)
        self.model = model

    def training_losses_seq2seq(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep in a sequence-to-sequence task.
        
        :param model: The transformer model.
        :param x_start: The initial data batch.
        :param t: The current timestep.
        :param model_kwargs: Additional keyword arguments for the model.
        :param noise: Optional Gaussian noise.
        :return: A dictionary with loss terms.
        """
        x_start_fix = x_start
        assert 'input_ids' in model_kwargs
        input_ids_x = model_kwargs.pop('input_ids')
        input_ids_mask = model_kwargs.pop('input_mask')
        x_start_mean = model.get_embeds(input_ids_x)

        std = self.sqrt_one_minus_alphas_cumprod[0].to(x_start_mean.device)
        x_start = self._get_x_start(x_start_mean, std)
        if noise is None:
            noise = th.randn_like(x_start)

        x_t = self.q_sample(x_start, t, noise=noise, mask=input_ids_mask)
        get_logits = model.get_logits

        terms = {}
        target = x_start
        model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)
        assert model_output.shape == target.shape == x_start.shape
        terms["mse"] = self.mean_flat((target - model_output) ** 2)

        model_out_x_start = self._x0_helper(model_output, x_t, t)['pred_xstart']
        t0_mask = (t == 0)
        t0_loss = self.mean_flat((x_start_mean - model_out_x_start) ** 2)
        terms["mse"] = th.where(t0_mask, t0_loss, terms["mse"])

        out_mean = self.q_mean_variance(x_start, th.LongTensor([self.num_timesteps - 1]).to(x_start.device))[0]
        tT_loss = self.mean_flat(out_mean ** 2)

        decoder_nll = self._token_discrete_loss(x_start, get_logits, input_ids_x)
        terms["nll"] = self._token_discrete_loss(model_out_x_start, get_logits, input_ids_x, mask=input_ids_mask, truncate=True, t=t)

        terms["loss"] = terms["mse"] + decoder_nll + tT_loss

        return terms

    def _get_x_start(self, x_start_mean, std):
        """
        Project word embeddings from {Emb(w)} to {x_0}.
        
        :param x_start_mean: Word embeddings.
        :param std: Standard deviation for noise.
        :return: x_0 with added noise.
        """
        noise = th.randn_like(x_start_mean)
        return x_start_mean + std * noise

    def _token_discrete_loss(self, x_t, get_logits, input_ids, mask=None, truncate=False, t=None):
        """
        Compute the discrete loss -log p(w|z_0).
        
        :param x_t: The current tensor.
        :param get_logits: Function to get logits from the model.
        :param input_ids: Input IDs for computing the loss.
        :param mask: Mask for the inputs.
        :param truncate: Whether to truncate the inputs.
        :param t: Current timestep.
        :return: The computed loss.
        """
        logits = get_logits(x_t)
        loss_fct = th.nn.CrossEntropyLoss(reduction='none')
        decoder_nll = loss_fct(logits.view(-1, logits.size(-1)), input_ids.view(-1)).view(input_ids.shape)
        if mask is not None:
            decoder_nll *= mask
        if mask is not None:
            decoder_nll = decoder_nll.sum(dim=-1) / mask.sum(dim=-1)
        else:
            decoder_nll = decoder_nll.mean(dim=-1)
        return decoder_nll

    def _x0_helper(self, model_output, x, t):
        """
        Helper function to compute x_0 from model output.
        
        :param model_output: Output from the model.
        :param x: The current tensor.
        :param t: Current timestep.
        :return: A dictionary with predicted x_prev and x_start.
        """
        pred_xstart = self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
        pred_prev, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
        return {'pred_xprev': pred_prev, 'pred_xstart': pred_xstart}

    def mean_flat(self, tensor):
        """
        Compute the mean over all non-batch dimensions.
        
        :param tensor: Input tensor.
        :return: Mean of the tensor.
        """
        return tensor.mean(dim=list(range(1, len(tensor.shape))))

