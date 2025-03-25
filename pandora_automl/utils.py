#!/usr/bin/env python3

from typing import Optional
import torch
from torch import Tensor
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.constraints.constraints import Interval

def fit_gp_model(
        X: Tensor, 
        objective_X: Tensor, 
        cost_X: Optional[Tensor] = None, 
        unknown_cost: bool = False, 
        kernel: Optional[torch.nn.Module] = None,
        gaussian_likelihood: bool = False,
        noisy_observation: bool = False, 
        noise_level: float = 1e-4,
        output_standardize: bool = False,
    ):
    # Ensure X is a 2D tensor [num_data, num_features]
    if X.ndim == 1:
        X = X.unsqueeze(dim=-1)
    
    # Ensure objective_X is a 2D tensor [num_data, 1]
    if objective_X.ndim == 1:
        objective_X = objective_X.unsqueeze(dim=-1)

    # Ensure cost_X is a 2D tensor [num_data, 1]
    if unknown_cost == True:
        if cost_X.ndim == 1:
            log_cost_X = torch.log(cost_X).unsqueeze(dim=-1)
        else:
            log_cost_X = torch.log(cost_X)

    Y = torch.cat((objective_X, log_cost_X), dim=-1) if unknown_cost else objective_X
        
    if noisy_observation:
        likelihood = None
    else:
        if gaussian_likelihood:
            _, aug_batch_shape = SingleTaskGP.get_batch_dimensions(
                    train_X=X,
                    train_Y=Y,
                )
            likelihood = GaussianLikelihood(
                    batch_shape=aug_batch_shape,
                    noise_constraint=Interval(lower_bound=noise_level, upper_bound=10*noise_level),
                )
        else:
            Yvar = torch.ones(len(Y)) * noise_level
            
            likelihood = FixedNoiseGaussianLikelihood(noise=Yvar)

    # Outcome transform
    if output_standardize == True:
        outcome_transform = Standardize(m=Y.shape[-1])
    else:
        outcome_transform = None
   
    model = SingleTaskGP(train_X=X, train_Y=Y, likelihood = likelihood, covar_module=kernel, outcome_transform=outcome_transform)

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    return model