# TODO explore other simplistic sample code
# https://github.com/lucidrains/denoising-diffusion-pytorch
# https://e-dorigatti.github.io/math/deep%20learning/2023/06/25/diffusion.html
# https://github.com/tanelp/tiny-diffusion
# NOTE adapted from diffuSeq, which is adapted from https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42
import numpy as np
import torch
import torch as th
import torch.nn as nn
import math
from tqdm.auto import tqdm  # For progress bar

class GaussianDiffusion():
    def __init__(self, betas, predict_xstart=True):
        self.predict_xstart = predict_xstart
        self.rescale_timesteps = True

        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    # NOTE the below comments from diffuSeq
    # self.mapping_func = None # implement in train main()
    # self.add_mask_noise = False # TODO

    # FIXME copied directly from diffuSeq
    def training_losses(self, model, *args, **kwargs):
        self.model = model
        return self.training_losses_seq2seq(model, *args, **kwargs)
    
    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        device = x_t.device
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape, device) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape, device) * eps
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        device = x_t.device
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape, device) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape, device)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t
    
    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        device = x_start.device
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape, device) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape, device)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape, device
        )
        return mean, variance, log_variance


    def q_sample(self, x_start, t, noise=None, mask=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :param mask: anchoring masked position
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        device = x_start.device
        x_t = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape, device) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape, device) * noise
        )

        mask = torch.broadcast_to(mask.unsqueeze(dim=-1), x_start.shape).to(device)
        return torch.where(mask==0, x_start, x_t)

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior: 
            q(x_{t-1} | x_t, x_0)

        """
        device = x_start.device
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape, device) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape, device) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape, device)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape, device
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_sample(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None,
        top_p=None, mask=None, x_start=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param mask: anchoring masked position to x_start
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        device = x.device
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if top_p is not None and top_p > 0:
            # print('top_p sampling')
            noise = th.randn_like(x)
            replace_mask = th.abs(noise) > top_p
            while replace_mask.any():
                noise[replace_mask] = th.randn_like(noise[replace_mask])
                replace_mask = th.abs(noise) > top_p
            assert (th.abs(noise) <= top_p).all()

        else:
            noise = th.randn_like(x)

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        if mask is None:
            pass
        else:
            sample = th.where(mask==0, x_start, sample)

        return {
            "sample": sample, 
            "pred_xstart": out["pred_xstart"],
            "greedy_mean": out["mean"], 
            "out": out
        }

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Compute the mean and variance for the diffusion posterior at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'mean': the posterior mean.
                 - 'variance': the posterior variance.
                 - 'log_variance': the log of the posterior variance.
                 - 'pred_xstart': the predicted x_0.
        """
        device = x.device
        if model_kwargs is None:
            model_kwargs = {}
        model_output = model(x, t, **model_kwargs).to(device)
        if denoised_fn is not None:
            model_output = denoised_fn(model_output)
        pred_xstart = self._predict_xstart_from_eps(x, t, model_output)
        if clip_denoised:
            pred_xstart = pred_xstart.clamp(-1, 1)

        posterior_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t
        )
        return {
            "mean": posterior_mean,
            "variance": posterior_variance,
            "log_variance": posterior_log_variance,
            "pred_xstart": pred_xstart,
        }

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        top_p=None,
        clamp_step=None,
        clamp_first=None,
        mask=None,
        x_start=None,
        gap=1,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param mask: anchoring masked position to x_start
        :param clamp_step: in clamp_first mode, choose end clamp step, otherwise starting clamp step
        :param clamp_first: bool, clamp_first mode
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = []
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            top_p=top_p,
            clamp_step=clamp_step,
            clamp_first=clamp_first,
            mask=mask,
            x_start=x_start
        ):
            final.append(sample['sample'])
        return final

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        top_p=None,
        clamp_step=None,
        clamp_first=None,
        mask=None,
        x_start=None,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None: # custom your the start point of x_0
            sample_x = noise
        else:
            sample_x = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices: # from T to 0
            t = th.tensor([i] * shape[0], device=device)
            if clamp_step is not None:
                if not clamp_first:
                    if i > clamp_step:
                        denoised_fn_cur = None
                    else:
                        denoised_fn_cur = denoised_fn
                else:
                    if i >= clamp_step:
                        denoised_fn_cur = denoised_fn
                    else:
                        denoised_fn_cur = None
            else:
                denoised_fn_cur = denoised_fn

            with th.no_grad():
                out = self.p_sample(
                    model,
                    sample_x,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn_cur,
                    model_kwargs=model_kwargs,
                    top_p=top_p,
                    mask=mask,
                    x_start=x_start
                )
                yield out
                sample_x = out["sample"]

    def _get_x_start(self, x_start_mean, std):
        '''
        Word embedding projection from {Emb(w)} to {x_0}
        :param x_start_mean: word embedding
        :return: x_0
        '''
        noise = th.randn_like(x_start_mean)
        assert noise.shape == x_start_mean.shape
        return x_start_mean + std * noise

    def _token_discrete_loss(self, x_t, get_logits, input_ids, mask=None, truncate=False, t=None):
        '''
        the loss of -log p(w|z_0)
        :param x_start_mean: word embedding
        :return: x_0
        '''
        reshaped_x_t = x_t
        logits = get_logits(reshaped_x_t)  # bsz, seqlen, vocab

        # Ensure input_ids is a LongTensor
        input_ids = input_ids.long()

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
        device = x.device
        pred_xstart = self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
        pred_prev, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
        return {'pred_xprev': pred_prev, 'pred_xstart': pred_xstart}

    def training_losses_seq2seq(self, model, x_start, t, model_kwargs=None, noise=None):
        device = x_start.device
        x_start_fix = x_start.to(device)

        input_ids_x = model_kwargs.pop('input_ids').long().to(device)
        input_ids_mask = model_kwargs.pop('input_mask').to(device)
        x_start_mean = model.get_embeds(input_ids_x).to(device)

        std = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, torch.tensor([0]).to(device), x_start_mean.shape, device)
        x_start = self._get_x_start(x_start_mean, std).to(device)

        if noise is None:
            noise = torch.randn_like(x_start).to(device)

        x_t = self.q_sample(x_start, t, noise=noise, mask=input_ids_mask).to(device)

        get_logits = model.get_logits

        terms = {}

        target = x_start
        model_output = model(x_t, self._scale_timesteps(t), **model_kwargs).to(device)
        terms["mse"] = mean_flat((target - model_output) ** 2).to(device)

        model_out_x_start = self._x0_helper(model_output, x_t, t)['pred_xstart'].to(device)
        t0_mask = (t == 0).to(device)
        t0_loss = mean_flat((x_start_mean - model_out_x_start) ** 2).to(device)
        terms["mse"] = terms["mse"].to(device)
        terms["mse"] = torch.where(t0_mask, t0_loss, terms["mse"])

        out_mean, _, _ = self.q_mean_variance(x_start, torch.LongTensor([self.num_timesteps - 1]).to(device))
        tT_loss = mean_flat(out_mean ** 2).to(device)

        decoder_nll = self._token_discrete_loss(x_start, get_logits, input_ids_x).to(device)
        terms["nll"] = self._token_discrete_loss(model_out_x_start, get_logits, input_ids_x, mask=input_ids_mask, truncate=True, t=t).to(device)

        terms["loss"] = (terms["mse"] + decoder_nll + tT_loss).to(device)

        return terms


    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
        langevin_fn=None,
        mask=None,
        x_start=None
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        device = x.device
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape, device)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape, device)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        if langevin_fn:
            sample = langevin_fn(sample, mean_pred, sigma, self.alphas_cumprod_prev[t[0]], t, x)
        
        if mask is None:
            pass
        else:
            sample = th.where(mask==0, x_start, sample)
        
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        device = x.device
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape, device) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape, device)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape, device)

        # Equation 12. reversed
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        top_p=None,
        clamp_step=None,
        clamp_first=None,
        mask=None,
        x_start=None,
        gap=1,
    ):
        """
        Generate samples from the model using DDIM.
        :param gap: compute ddim sampling for each {gap} step

        Same usage as p_sample_loop().
        """
        final = []
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            mask=mask,
            x_start=x_start,
            gap = gap
        ):
            final.append(sample['sample'])
        return final

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        langevin_fn=None,
        mask=None,
        x_start=None,
        gap=1
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            sample_x = noise
        else:
            sample_x = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1][::gap]

        if progress:
            # Lazy import so that we don't depend on tqdm.

            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    sample_x,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    mask=mask,
                    x_start=x_start
                )
                yield out
                sample_x = out["sample"]

def _extract_into_tensor(arr, timesteps, broadcast_shape, device):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :param device: the device to move the tensor to.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))
