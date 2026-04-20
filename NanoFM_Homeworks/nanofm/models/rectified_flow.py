"""
Rectified Flow - Exercise Implementation
=========================================

Fill in the code sections marked with ??? to implement Rectified Flow.

Rectified Flow defines a straight-line interpolation between data (x) and noise (z1):
    z_t = (1 - t) * x + t * z1

A neural network v_theta(z_t, t) is trained to predict the velocity field:
    v = z1 - x

During sampling, we solve the ODE backwards from noise to data using the Euler method:
    z_{t-dt} = z_t - dt * v_theta(z_t, t)
"""

import torch


class RectifiedFlow:
    """
    Implements Rectified Flow for generative modeling.

    Key concepts:
    - Forward process: straight-line interpolation between data and noise
    - Velocity field: the neural network predicts the direction from data to noise
    - Sampling: Euler integration of the learned ODE from noise back to data
    """

    def __init__(self, model, ln=True):
        """
        Args:
            model: Neural network v_theta(z_t, t, cond) that predicts the velocity field.
                   Takes (noisy_input, timestep, condition) and returns predicted velocity.
            ln: If True, use logit-normal distribution for sampling timesteps during training.
                If False, use uniform distribution U(0, 1).
        """
        self.model = model
        self.ln = ln

    def forward(self, x, cond):
        """
        Compute the Rectified Flow training loss.

        The training procedure is:
        1. Sample random timesteps t for each sample in the batch
        2. Sample noise z1 ~ N(0, I)
        3. Compute interpolated samples: z_t = (1 - t) * x + t * z1
        4. Predict velocity: v_theta = model(z_t, t, cond)
        5. Compute MSE loss between predicted and true velocity: ||v_theta - (z1 - x)||^2

        Args:
            x: Clean data tensor of shape (B, C, H, W)
            cond: Conditioning tensor (e.g., class labels) of shape (B,)

        Returns:
            loss: Scalar mean training loss
            ttloss: List of (timestep_value, loss_value) tuples for per-timestep logging
        """
        b = x.size(0)

        # ============================================================
        # Exercise 7.1: Sample timesteps
        # ============================================================
        # Use logit-normal (self.ln=True) or uniform (self.ln=False) sampling.
        # Ensure t has shape (b,) and is on the same device as x.
        if self.ln:
            nt = torch.randn(b, device=x.device)
            t = torch.sigmoid(nt)
        else:
            t = torch.rand(b, device=x.device)

        # Reshape t for broadcasting with spatial dimensions: (b,) -> (b, 1, 1, 1)
        texp = t.view([b, *([1] * len(x.shape[1:]))])

        # ============================================================
        # Exercise 7.2: Compute the interpolated (noisy) samples z_t.
        # ============================================================
        # Use texp (the reshaped t) for broadcasting with spatial dimensions.
        z1 = torch.randn_like(x)
        zt = (1 - texp) * x + texp * z1

        # ============================================================
        # Exercise 7.3: Predict velocity and compute MSE loss.
        # ============================================================
        # The target is the straight-line velocity from data to noise.
        # Average the squared error over spatial dimensions, keeping the batch dimension.
        vtheta = self.model(zt, t, cond)
        batchwise_mse = (vtheta - (z1 - x)).pow(2).mean(dim=(1, 2, 3))

        # Logging: record per-timestep losses (no need to modify this)
        tlist = batchwise_mse.detach().cpu().reshape(-1).tolist()
        ttloss = [(tv, tloss) for tv, tloss in zip(t, tlist)]
        return batchwise_mse.mean(), ttloss

    @torch.no_grad()
    def sample(self, z, cond, null_cond=None, sample_steps=50, cfg=2.0):
        """
        Generate samples by solving the ODE from noise to data using the Euler method.

        Starting from z_1 ~ N(0, I), we iterate backwards in time:
            z_{t-dt} = z_t - dt * v_theta(z_t, t)

        If null_cond is provided, classifier-free guidance (CFG) is applied:
            v_guided = v_uncond + cfg * (v_cond - v_uncond)

        Args:
            z: Initial noise tensor of shape (B, C, H, W)
            cond: Conditioning tensor (e.g., class labels) of shape (B,)
            null_cond: Null/unconditional labels for CFG. If None, no CFG is applied.
            sample_steps: Number of Euler integration steps
            cfg: Classifier-free guidance scale (higher = more class-typical, less diverse)

        Returns:
            images: List of intermediate z tensors showing the generation trajectory.
                    images[0] is the initial noise, images[-1] is the final generated sample.
        """
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        images = [z]

        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(z.device)

            # ============================================================
            # Exercise 11.1: Get the conditional velocity prediction.
            # ============================================================
            vc = self.model(z, t, cond)

            # ============================================================
            # Exercise 11.2: Apply classifier-free guidance (if null_cond is given).
            # ============================================================
            # Refer to Section 10 in the notebook for the CFG formula:
            #     v_guided = v_uncond + cfg * (v_cond - v_uncond)
            if null_cond is not None:
                vu = self.model(z, t, null_cond)
                vc = vu + cfg * (vc - vu)

            # ============================================================
            # Exercise 11.3: Euler integration step.
            # ============================================================
            z = z - dt * vc

            images.append(z)

        return images

