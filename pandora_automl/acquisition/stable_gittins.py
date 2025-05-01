#!/usr/bin/env python3

# Copyright (c) 2024 Qian Xie
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from typing import Callable, Optional, Union
import torch
from torch import Tensor
from torch.autograd import (Function, grad)
from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.models.model import Model
from botorch.acquisition.analytic import (_scaled_improvement, _log_ei_helper)
from botorch.acquisition.objective import PosteriorTransform
from botorch.utils.transforms import t_batch_mode_transform


# Set default tensor type to float64
torch.set_default_dtype(torch.float64)


class StableGittinsIndex(AnalyticAcquisitionFunction):
    r"""Single-outcome/Two-outcome Gittins Index (analytic).

    Computes Gittins Index using the bisection search and the analytic formula of 
    LogExpectedImprovement for a Normal posterior distribution, in a numerically robust manner. 
    In particular, the implementation takes special care to avoid numerical issues in the 
    computation of the acquisition value and its gradient in regions where improvement is 
    predicted to be virtually impossible. Only supports the case of `q=1`. The model can be 
    either single-outcome or two-outcome.

    `StablePBGI(x) = argmin_g |log(E(max(f(x) - g, 0))) - log(lmbda) - log(c(x))|,`

    where the expectation is taken over the value of stochastic function `f` at `x`.

    Example:
        Uniform-cost:
        >>> model = fit_gp_model(train_X, train_Y)
        >>> StablePBGI = StableGittinsIndex(model, lmbda=0.0001)
        >>> pbgi = StablePBGI(test_X)
        
        Varying-cost:
        >>> def cost_function(x):
        >>>     return 1+20*x.mean(dim=-1))
        >>> model = fit_gp_model(train_X, train_Y)
        >>> StablePBGI = StableGittinsIndex(model, lmbda=0.0001, cost=cost_function)
        >>> pbgi = StablePBGI(test_X)

        Unknown-cost:
        >>> model = fit_gp_model(train_X, train_Y, cost_X)
        >>> StablePBGI = StableGittinsIndex(model, lmbda=0.0001, unknown_cost=True)
        >>> pbgi = StablePBGI(test_X)
    """

    def __init__(
        self,
        model: Model,
        lmbda: float,
        posterior_transform: Optional[PosteriorTransform] = None,
        maximize: bool = True,
        bound: torch.Tensor = torch.tensor([[-1.0], [1.0]], dtype=torch.float64),
        eps: float = 1e-6,
        cost: Optional[Callable] = None,
        unknown_cost: bool = False,
    ):
        r"""Single-outcome/Two-outcome Gittins Index (analytic).
        
        Args:
            model: A fitted single-outcome model or a fitted two-outcome model, 
                where the first output corresponds to the objective 
                and the second one to the log-cost.
            lmbda: A scalar representing the cost-per-sample or the scaling factor of the cost function.
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            maximize: If True, consider the problem a maximization problem.
            cost: A callable cost function. If None, consider the problem a uniform-cost problem.
            unknown_cost: If True, consider the problem an unknown-cost problem.
            bound: A `2 x d` tensor of lower and upper bound for each column of `X`.
        """
        # use AcquisitionFunction constructor to avoid check for objective
        super(AnalyticAcquisitionFunction, self).__init__(model=model)
        self.lmbda = lmbda
        self.posterior_transform = posterior_transform
        self.maximize = maximize
        self.bound = bound
        self.eps = eps
        self.cost = cost if cost is not None else 1.0
        self.unknown_cost = unknown_cost      
        
    @t_batch_mode_transform(expected_q=1, assert_output_shape=False)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate Gittins Index on the candidate set X using bisection method.

        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.
                Gittins Index is computed for each point individually,
                i.e., what is considered are the marginal posteriors, not the joint.

        Returns:
            A `(b1 x ... bk)`-dim tensor of Gittins Index values at the
            given design points `X`.
        """
        
        if self.unknown_cost:
            # Handling the unknown cost scenario
            posterior = self.model.posterior(X)
            means = posterior.mean  # (b) x 2
            vars = posterior.variance.clamp_min(1e-6)  # (b) x 2
            stds = vars.sqrt()

            mean_obj = means[..., 0].squeeze(dim=-1)
            std_obj = stds[..., 0].squeeze(dim=-1)

            log_mgf = (means[..., 1] + 0.5 * vars[..., 1]).squeeze(dim=-1)

            gi_value = StableGittinsIndexFunction.apply(X, mean_obj, std_obj, self.lmbda, self.maximize, self.bound, self.eps, log_mgf)

        else:
            # Handling the known cost scenario
            mean, sigma = self._mean_and_sigma(X)

            if callable(self.cost):
                log_cost_X = self.cost(X).view(mean.shape).log()
            else:
                log_cost_X = torch.zeros_like(mean)

            gi_value = StableGittinsIndexFunction.apply(X, mean, sigma, self.lmbda, self.maximize, self.bound, self.eps, log_cost_X)

        # If maximizing, return the GI value as is; if minimizing, return its negative
        return gi_value
    

class StableGittinsIndexFunction(Function):
    @staticmethod
        
    def forward(
        ctx, 
        X: Tensor, 
        mean: Tensor, 
        sigma: Tensor, 
        lmbda: float, 
        maximize: bool, 
        bound: Tensor, 
        eps: float, 
        log_cost_X: Union[float, Tensor],
    ):

        def cost_adjusted_log_expected_improvement(best_f):
            u = _scaled_improvement(mean, sigma, best_f, maximize)
            return _log_ei_helper(u) + sigma.log() - torch.log(torch.tensor(lmbda)) - log_cost_X


        size = X.size(0)
        l = bound[0] * torch.ones(size, requires_grad=False)
        h = bound[1] * torch.ones(size, requires_grad=False)

        if maximize:
            while torch.any(cost_adjusted_log_expected_improvement(best_f=l) < 0):
                l = 2 * l
            while torch.any(cost_adjusted_log_expected_improvement(best_f=h) > 0):
                h = 2 * h
        else:
            while torch.any(cost_adjusted_log_expected_improvement(best_f=l) > 0):
                l = 2 * l
            while torch.any(cost_adjusted_log_expected_improvement(best_f=h) < 0):
                h = 2 * h

        # Bisection method
        m = (h + l) / 2
        for i in range(100):
            sgn_m = torch.sign(cost_adjusted_log_expected_improvement(best_f=m))
            if maximize:
                l = torch.where(sgn_m >= 0, m, l)
                h = torch.where(sgn_m <= 0, m, h)
            else:
                l = torch.where(sgn_m <= 0, m, l)
                h = torch.where(sgn_m >= 0, m, h)
            m = (h + l) / 2

        # Save u for backward computation
        u = _scaled_improvement(mean, sigma, m, maximize)
        
        # Save values needed in the backward pass
        ctx.save_for_backward(X, mean, sigma, u, log_cost_X)
        
        # Save boolean flag directly in ctx
        ctx.maximize = maximize
            
        return m
    
    @staticmethod
    def backward(ctx, grad_output):
                
        # Retrieve saved tensors
        X, mean, sigma, u, log_cost_X = ctx.saved_tensors
        maximize = ctx.maximize  # Retrieve the boolean flag directly from ctx
                
        # Gradient of the mean function with respect to X
        dmean_dX = grad(outputs=mean, inputs=X, grad_outputs=torch.ones_like(mean), retain_graph=True, allow_unused=True)[0]

        # Gradient of the std function with respect to X
        dsigma_dX = grad(outputs=sigma, inputs=X, grad_outputs=torch.ones_like(sigma), retain_graph=True, allow_unused=True)[0]

        # Gradient of the log_h function with respect to u
        with torch.enable_grad():
            u.requires_grad_()
            log_h_u = _log_ei_helper(u)
            dlogh_du = grad(
                outputs=log_h_u,
                inputs=u,
                grad_outputs=torch.ones_like(log_h_u),
                retain_graph=True,
                allow_unused=True,
            )[0]

        if log_cost_X.requires_grad:
            # Compute gradient only if cost_X is not a scalar
            dlog_cost_dX = grad(outputs=log_cost_X, inputs=X, grad_outputs=torch.ones_like(log_cost_X), retain_graph=True, allow_unused=True)[0]
        else:
            # If cost_X does not require grad, set its gradient to zero
            dlog_cost_dX = torch.zeros_like(X)

        # Check if gradients are None and handle accordingly
        if dmean_dX is None or dsigma_dX is None or dlog_cost_dX is None or dlogh_du is None:
            raise RuntimeError("Gradients could not be computed for one or more components.")
        
        # Compute the gradient of the Stable Gittins Index acquisition function
        if maximize:
            grad_X = grad_output.unsqueeze(-1).unsqueeze(-1) * (dmean_dX - dsigma_dX * u.unsqueeze(-1).unsqueeze(-1) + dsigma_dX / dlogh_du.unsqueeze(-1).unsqueeze(-1) - sigma.unsqueeze(-1).unsqueeze(-1) * dlog_cost_dX)
        else:
            grad_X = grad_output.unsqueeze(-1).unsqueeze(-1) * (dmean_dX + dsigma_dX * u.unsqueeze(-1).unsqueeze(-1) - dsigma_dX / dlogh_du.unsqueeze(-1).unsqueeze(-1) - sigma.unsqueeze(-1).unsqueeze(-1) * dlog_cost_dX)
        
        return grad_X, None, None, None, None, None, None, None