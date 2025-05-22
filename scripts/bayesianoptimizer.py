#!/usr/bin/env python3

from typing import Callable, Optional
import torch
from torch import Tensor
from botorch.acquisition import ExpectedImprovement, LogExpectedImprovement, UpperConfidenceBound
from botorch.generation.gen import gen_candidates_torch
from botorch.acquisition import PosteriorMean
from pandora_automl.acquisition.lcb import LowerConfidenceBound
from pandora_automl.acquisition.log_ei import LogVanillaExpectedImprovement, StableExpectedImprovement
from botorch.acquisition.multi_step_lookahead import warmstart_multistep
from pandora_automl.acquisition.gittins import GittinsIndex
from pandora_automl.acquisition.stable_gittins import StableGittinsIndex
from pandora_automl.acquisition.ei_puc import ExpectedImprovementWithCost
from pandora_automl.acquisition.log_ei_puc import LogExpectedImprovementWithCost
from pandora_automl.acquisition.multi_step_ei import MultiStepLookaheadEI
from pandora_automl.acquisition.budgeted_multi_step_ei import BudgetedMultiStepLookaheadEI
from botorch.sampling.pathwise import draw_matheron_paths
from botorch.utils.sampling import optimize_posterior_samples
from botorch.acquisition.predictive_entropy_search import qPredictiveEntropySearch
from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy,qMultiFidelityMaxValueEntropy
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.models.deterministic import GenericDeterministicModel
from botorch.optim import optimize_acqf
from scipy.stats import norm
from copy import copy
from pandora_automl.utils import fit_gp_model
import numpy as np
import math
import time

class BayesianOptimizer:
    DEFAULT_COST = torch.tensor(1.0)  # Default cost if not provided

    def __init__(self,
                 dim: int, 
                 maximize: bool, 
                 initial_points: Tensor, 
                 objective: Optional[Callable] = None, 
                 cost: Optional[Callable] = None, 
                 objective_cost: Optional[Callable] = None, 
                 kernel: Optional[torch.nn.Module] = None,
                 noisy_observation: bool = False,
                 noise_level: Optional[float] = 0.1,
                 output_standardize: bool = False,
                ):
        self.validate_functions(objective, objective_cost)
        self.initialize_attributes(objective, cost, objective_cost, dim, maximize, initial_points, kernel, noisy_observation, noise_level, output_standardize)


    def validate_functions(self, objective, objective_cost):
        # Make sure that the objective function and the cost function are passed in the correct form
        if objective_cost is None and objective is None:
            raise ValueError("At least one of 'objective' or 'objective_cost' must be provided.")
        if objective is not None and objective_cost is not None:
            raise ValueError("Only one of 'objective' or 'objective_cost' can be provided.")
        self.unknown_cost = callable(objective_cost)


    def initialize_attributes(self, objective, cost, objective_cost, dim, maximize, initial_points, kernel, noisy_observation, noise_level, output_standardize):
        self.objective = objective
        self.cost = cost if cost is not None else self.DEFAULT_COST
        self.objective_cost = objective_cost
        self.dim = dim
        self.maximize = maximize
        # need to be initialized after self.dim
        self.bounds = torch.stack([torch.zeros(dim), torch.ones(dim)])
        self.best_f = None
        self.best_x = None 
        self.best_history = []
        self.cumulative_cost = 0.0
        self.cost_history = [0.0]
        self.acq_history = [np.nan]
        self.stopping_history = {}
        self.runtime_history = []
        self.lmbda_history = []
        # GP model parameters
        self.kernel = kernel
        self.noisy_observation = noisy_observation
        self.noise_level = noise_level
        self.output_standardize = output_standardize
        self.suggested_x_full_tree = None
        self.model = None
        self.old_model = None
        self.num_samples = 64
        self.granularity = 10001
        self.mask = torch.ones(self.granularity, dtype=torch.bool)
        self.iteration = 0
        self.initialize_points(initial_points)
    

    def initialize_points(self, initial_points):
        self.x = initial_points
        if callable(self.objective):
            self.y = self.objective(initial_points)
            if callable(self.cost):
                self.c = self.cost(initial_points).view(-1)
            else:
                self.c = self.DEFAULT_COST
        if callable(self.objective_cost):
            self.y, self.c = self.objective_cost(initial_points)
        if self.noisy_observation:
            noise = torch.randn_like(self.y) * self.noise_level
            self.y += noise

        if self.dim == 1:
            for point in initial_points:
                self.mask[int(point.detach()*(self.granularity - 1))] = False
        self.update_best()


    def update_best(self):
        self.best_f = self.y.max().item() if self.maximize else self.y.min().item()
        self.best_x = self.x[self.y.argmax()] if self.maximize else self.x[self.y.argmin()]
        self.best_history.append(self.best_f)
    
    def iterate(self, acquisition_function_class, **acqf_kwargs):

        is_ts = False
        is_pes = False
        gaussian_likelihood = False
    
        if acquisition_function_class in (ExpectedImprovementWithCost, LogExpectedImprovementWithCost, GittinsIndex, BudgetedMultiStepLookaheadEI):
            unknown_cost = self.unknown_cost
        else:
            unknown_cost = False

        self.old_model = self.model
        model = fit_gp_model(
            X=self.x.detach(), 
            objective_X=self.y.detach(), 
            cost_X=self.c.detach(), 
            unknown_cost=unknown_cost,
            kernel=self.kernel,
            gaussian_likelihood=gaussian_likelihood,
            output_standardize=self.output_standardize,
        )
        self.model = model
        if (self.old_model is None):
            self.old_model = model

        acqf_args = {'model': model}
        
        if acquisition_function_class == "ThompsonSampling":
        
            # Draw sample path(s)
            paths = draw_matheron_paths(model, sample_shape=torch.Size([1]))
            
            # Optimize
            optimal_input, optimal_output = optimize_posterior_samples(paths=paths, bounds=self.bounds, maximize=self.maximize)

            is_ts = True
            new_point = optimal_input
            self.current_acq = optimal_output.item()
        
        if acquisition_function_class in (GittinsIndex, StableGittinsIndex):
            acqf_args['maximize'] = self.maximize
            
            if acqf_kwargs.get('step_EIpu') == True:
                if self.need_lmbda_update:
                    if callable(self.cost) or callable(self.objective_cost):
                        # Optimize EIpu first to get new_point_EIpu
                        EIpu = ExpectedImprovementWithCost(model=model, best_f=self.best_f, maximize=self.maximize, cost=self.cost, unknown_cost=self.unknown_cost)
                        _, new_point_EIpu = optimize_acqf(
                            acq_function=EIpu,
                            bounds=self.bounds,
                            q=1,
                            num_restarts=10*self.dim,
                            raw_samples=200*self.dim,
                            options={'method': 'L-BFGS-B'},
                        )
                        if self.current_lmbda == None:
                            self.current_lmbda = new_point_EIpu.item() / 2
                        else:
                            self.current_lmbda = min(self.current_lmbda, new_point_EIpu.item() / 2)

                    else:
                        # Optimize EI first to get new_point_EI
                        EI = ExpectedImprovement(model=model, best_f=self.best_f, maximize=self.maximize)
                        _, new_point_EI = optimize_acqf(
                            acq_function=EI,
                            bounds=self.bounds,
                            q=1,
                            num_restarts=10*self.dim,
                            raw_samples=200*self.dim,
                            options={'method': 'L-BFGS-B'},
                        )
                        if self.current_lmbda == None:
                            self.current_lmbda = new_point_EI.item() / 2
                        else:
                            self.current_lmbda = min(self.current_lmbda, new_point_EI.item() / 2)
                    self.need_lmbda_update = False  # Reset the flag
                
                acqf_args['lmbda'] = self.current_lmbda
                self.lmbda_history.append(self.current_lmbda)
                
            elif acqf_kwargs.get('step_divide') == True:
                if self.need_lmbda_update:
                    self.current_lmbda = self.current_lmbda / acqf_kwargs.get('alpha')
                    self.need_lmbda_update = False
                acqf_args['lmbda'] = self.current_lmbda
                self.lmbda_history.append(self.current_lmbda)

            else: 
                acqf_args['lmbda'] = acqf_kwargs['lmbda']
                self.lmbda_history.append(acqf_kwargs['lmbda'])

            acqf_args['cost'] = self.cost
            acqf_args['unknown_cost'] = self.unknown_cost

        
        elif acquisition_function_class == UpperConfidenceBound:
            if acqf_kwargs.get('heuristic') == True:
                print("Using heuristic for beta")
                acqf_args['beta'] = 2*np.log(self.dim*((self.cumulative_cost+1)**2)*(math.pi**2)/(6*0.1))/5
            else:
                acqf_args['beta'] = acqf_kwargs['beta']
            acqf_args['maximize'] = self.maximize
        
        
        elif acquisition_function_class in (ExpectedImprovement, LogExpectedImprovement, LogVanillaExpectedImprovement, StableExpectedImprovement):
            acqf_args['best_f'] = self.best_f
            acqf_args['maximize'] = self.maximize

        
        elif acquisition_function_class in (ExpectedImprovementWithCost, LogExpectedImprovementWithCost):
            acqf_args['best_f'] = self.best_f
            acqf_args['maximize'] = self.maximize
            acqf_args['cost'] = self.cost
            acqf_args['unknown_cost'] = self.unknown_cost
            if acqf_kwargs.get('cost_cooling') == True:
                cost_exponent = (self.budget - self.cumulative_cost) / self.budget
                cost_exponent = max(cost_exponent, 0)  # Ensure cost_exponent is non-negative
                acqf_args['cost_exponent'] = cost_exponent

        else:
            acqf_args.update(**acqf_kwargs)


        if is_ts == False and is_pes == False:
            acq_function = acquisition_function_class(**acqf_args)

            # use grid search if dimension is 1, otherwise use optimization
            if self.dim == 1:
                print("Using grid search for 1D optimization")
                candidates = torch.linspace(0, 1, self.granularity).unsqueeze(1).unsqueeze(1)
                candidates_acq_vals = acq_function.forward(candidates[self.mask])
                candidates =  candidates.detach()
                # change to reflect minimization objective
                if (self.maximize):
                    best_idx = torch.argmax(candidates_acq_vals.view(-1), dim=0)
                else:
                    best_idx = torch.argmin(candidates_acq_vals.view(-1), dim=0)
                best_point = candidates[best_idx]
                best_acq_val = candidates_acq_vals[best_idx].item()

            else:
                print("Using optimization for multi-dimensional optimization")
                candidate, candidate_acq_val = optimize_acqf(
                    acq_function=acq_function,
                    bounds=torch.stack([torch.zeros(self.dim), torch.ones(self.dim)]),
                    q=1,
                    num_restarts=10*self.dim,
                    raw_samples=1024*self.dim,
                    gen_candidates=gen_candidates_torch
                )
                best_point = candidate.detach()
                best_acq_val = candidate_acq_val.item() 

            new_point = best_point
            self.current_acq = best_acq_val


        if self.unknown_cost:
            new_value, new_cost = self.objective_cost(new_point.detach())
        else: 
            new_value = self.objective(new_point.detach())

        if self.noisy_observation:
            noise = torch.randn_like(new_value) * self.noise_level
            new_value += noise

        self.x = torch.cat((self.x, new_point.detach()))
        self.y = torch.cat((self.y, new_value.detach()))
        
        # Record statistics about different stopping rules
        self.log_time(self.update_stopping_criteria, "PRB", skip_prb=False)
        self.log_time(self.update_stopping_criteria, "StablePBGI", lmbda=0.1)
        self.log_time(self.update_stopping_criteria, "StablePBGI", lmbda=0.01)
        self.log_time(self.update_stopping_criteria, "StablePBGI", lmbda=0.001)
        self.log_time(self.update_stopping_criteria, "LogEIC")
        self.log_time(self.update_stopping_criteria, "UCB-LCB")
        self.log_time(self.update_stopping_criteria, "Expected_Min_Regret_Gap")


        self.update_best()
        self.update_cost(new_point)
        print("New point:", new_point.detach())
        if (self.dim == 1):
            self.mask[int(new_point.detach()*(self.granularity - 1))] = False
        self.iteration += 1

        self.acq_history.append(self.current_acq)

        # Check if lmbda needs to be updated in the next iteration
        if acquisition_function_class in (GittinsIndex, StableGittinsIndex) and (acqf_kwargs.get('step_EIpu') == True or acqf_kwargs.get('step_divide') == True):
            if (self.maximize and self.current_acq < self.best_f) or (not self.maximize and -self.current_acq > self.best_f):
                self.need_lmbda_update = True

    def if_not_exist_create_key(self, key):
        if key not in self.stopping_history:
            self.stopping_history[key] = [np.nan]  # initialize if missing

    def log_time(self, func, *args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        print(f"{func.__name__}({args[0] if args else ''}) took {elapsed:.4f} s")

        # --- record the timing in `stopping_history` ---------------------------
        if func.__name__ == "update_stopping_criteria":
            crit = args[0]                                    # e.g. "PRB", "StablePBGI"
            if crit == "StablePBGI":
                crit_key = f"StablePBGI({kwargs.get('lmbda', 0.01)})"
            elif crit == "PRB":                               # keep in sync with ε = 0.1
                crit_key = "PRB_0.1"
            else:
                crit_key = crit

            time_key = f"{crit_key}_time"
            self.if_not_exist_create_key(time_key) 
            self.stopping_history[time_key].append(elapsed)
        # -----------------------------------------------------------------------
        return result


    def update_stopping_criteria(self, stopping_criteria, lmbda=0.01, skip_prb=True):
        '''
        This function implements the following stopping rules: 
                                        'StablePBGI(1e-1)',
                                        'StablePBGI(1e-2)',
                                        'StablePBGI(1e-3)',
                                        'LogEIC',
                                        'regret upper bound',
                                        'exp min regret gap',
                                        'PRB'
        '''
        # Currently only works for dim=1
        # Initialization for probabilistic regret bound (PRB) stopping rule
        epsilon = 0.1
        candidates = torch.linspace(0, 1, self.granularity).unsqueeze(1)

        if (stopping_criteria == "PRB"):
            # Probabilistic regret bound
            key = f'PRB_{epsilon}'
            self.if_not_exist_create_key(key)
            paths = draw_matheron_paths(self.model, sample_shape=torch.Size([self.num_samples]))
            bounds = torch.stack([torch.zeros(self.dim), torch.ones(self.dim)])
            
            # When to not skip PRB calculation
            # 1. If the iteration is less than 50 and the iteration is a multiple of 5
            # 2. If the iteration is a multiple of 10
            if (self.dim > 1 and skip_prb==True):
                skip_PRB = not ((self.iteration < 50 and self.iteration % 5 == 0) or self.iteration % 10 == 0)
                if skip_PRB:
                    print("Skipping PRB calculation")
                    self.if_not_exist_create_key(key)
                    self.stopping_history[key].append(self.stopping_history[key][-1])
                    return
        
            maximize_factor = 1 if self.maximize else -1
            if (self.dim == 1):
                regrets = (maximize_factor*paths(candidates)).max(dim=1).values - maximize_factor*paths(self.best_x.unsqueeze(0)).squeeze(-1)
            else:
                # 1. build a QMC sampler that will internally draw your fantasy paths
                _, optimum_values = optimize_posterior_samples(paths=paths, 
                                                               bounds=bounds, 
                                                               raw_samples=200*self.dim, 
                                                               num_restarts=10*self.dim,
                                                               maximize=self.maximize)
                # print("optimum_values:", optimum_values)
                regrets = maximize_factor*optimum_values.squeeze(-1) - maximize_factor*paths(self.best_x.unsqueeze(0)).squeeze(-1)
                # print("regrets:", regrets)
            
            prb_estimate = (regrets <= epsilon).float().mean().item()
            self.stopping_history[key].append(prb_estimate)
            
                # print("Probabilistic regret bound")
                # print(f'Epsilon: {epsilon}, regrets: {prb_estimate}, num_samples: {self.num_samples}')
            self.num_samples = min(math.ceil(self.num_samples * 1.5), 1000)
 
        elif (stopping_criteria == "StablePBGI"):
            # 3. Stable PBGI
            key = f'StablePBGI({lmbda})'
            self.if_not_exist_create_key(key) 
            StablePBGI = StableGittinsIndex(model=self.model, maximize=self.maximize, lmbda=lmbda, cost=self.cost)
            maximize_factor = 1 if self.maximize else -1
            if (self.dim == 1): 
                StablePBGI_acq = StablePBGI.forward(candidates.unsqueeze(1))
                new_config_acq = maximize_factor*torch.max(maximize_factor*StablePBGI_acq[self.mask]) 
                
            else:
                NegStablePBGI = lambda x: -StablePBGI(x)
                SignedStablePBGI = StablePBGI if self.maximize else NegStablePBGI
                # Optimize the acquisition function
                candidates, StablePBGI_acq = optimize_acqf(
                    acq_function=SignedStablePBGI,
                    bounds=self.bounds,
                    q=1,
                    num_restarts=10*self.dim,
                    raw_samples=1024*self.dim,
                    gen_candidates=gen_candidates_torch
                )
    
                new_config_acq = maximize_factor*StablePBGI_acq 

            self.stopping_history[key].append(new_config_acq.item())
            # print("StablePBGI")
            # print(f'Lambda: {lmbda}, acquisition: {new_config_acq.item()}') 

        elif (stopping_criteria == "LogEIC"):
            key = 'LogEIC'
            self.if_not_exist_create_key(key)
            LogEIC = LogExpectedImprovementWithCost(model=self.model, best_f=self.best_f, maximize=self.maximize, cost=self.cost)
            # maximization or minimization objective has no effect on LogEIC
            if (self.dim == 1):
                LogEIC_acq = LogEIC.forward(candidates.unsqueeze(1)) 
                new_config_acq = torch.max(LogEIC_acq[self.mask]).detach()
            else:
                candidates, LogEIC_acq = optimize_acqf(
                        acq_function=LogEIC,
                        bounds=self.bounds,
                        q=1,
                        num_restarts=10*self.dim,
                        raw_samples=1024*self.dim,
                        gen_candidates=gen_candidates_torch
                    )
                new_config_acq = LogEIC_acq # torch.max(LogEIC_acq)
        
            self.stopping_history[key].append(new_config_acq.item())

        elif (stopping_criteria == "UCB-LCB"):  
            
            key = f'UCB-LCB'
            self.if_not_exist_create_key(key) 

            UCB = UpperConfidenceBound(model=self.model, maximize=self.maximize, beta=2 * np.log(self.dim * ((self.iteration + 1) ** 2) * (math.pi ** 2) / (6 * 0.1)) / 5)
            LCB = LowerConfidenceBound(model=self.model, maximize=self.maximize, beta=2 * np.log(self.dim * ((self.iteration + 1) ** 2) * (math.pi ** 2) / (6 * 0.1)) / 5)
            if (self.maximize):
                optimistic_CB = UCB; pessimistic_CB = LCB; maximize_factor = 1
            else:
                optimistic_CB = LCB; pessimistic_CB = UCB; maximize_factor = -1
            # print(f"beta: {beta}")
            if (self.dim == 1):
                optimistic_acq = optimistic_CB.forward(candidates.unsqueeze(1))
                pessimistic_acq = pessimistic_CB.forward(self.x.unsqueeze(1))
                kappa = torch.max(maximize_factor*optimistic_acq) - torch.max(maximize_factor*pessimistic_acq)
            else:
                candidates, optimistic_acq = optimize_acqf(
                        acq_function=optimistic_CB,
                        bounds=self.bounds,
                        q=1,
                        num_restarts=10*self.dim,
                        raw_samples=1024*self.dim,
                        gen_candidates=gen_candidates_torch
                    ) 

                # print(candidates.shape)
                pessimistic_acq = pessimistic_CB.forward(self.x.unsqueeze(1))
                # print(f"UCB {UCB_acq} and LCB {torch.max(LCB_acq)}")
                
                kappa = maximize_factor*optimistic_acq - torch.max(maximize_factor*pessimistic_acq)
                # print(f"kappa: {kappa}")
                # kappa = torch.max(UCB_acq) - torch.max(LCB_acq)
            self.stopping_history[key].append(kappa.item())
        
        elif (stopping_criteria == "Expected_Min_Regret_Gap"): 
            key = f'Expected-Min-Regret-Gap'
            self.if_not_exist_create_key(key) 
            
            UCB = UpperConfidenceBound(model=self.model, maximize=self.maximize, beta=2 * np.log(self.dim * ((self.iteration + 1) ** 2) * (math.pi ** 2) / (6 * 0.1)) / 5)
            LCB = LowerConfidenceBound(model=self.model, maximize=self.maximize, beta=2 * np.log(self.dim * ((self.iteration + 1) ** 2) * (math.pi ** 2) / (6 * 0.1)) / 5)
            if (self.maximize):
                optimistic_CB = UCB; pessimistic_CB = LCB; maximize_factor = 1
            else:
                optimistic_CB = LCB; pessimistic_CB = UCB; maximize_factor = -1
            # print(f"beta: {beta}")
            if (self.dim == 1):
                optimistic_acq = optimistic_CB.forward(candidates.unsqueeze(1))
                pessimistic_acq = pessimistic_CB.forward(self.x.unsqueeze(1))
                kappa = torch.max(maximize_factor*optimistic_acq) - torch.max(maximize_factor*pessimistic_acq)
            else:
                candidates, optimistic_acq = optimize_acqf(
                        acq_function=optimistic_CB,
                        bounds=self.bounds,
                        q=1,
                        num_restarts=10*self.dim,
                        raw_samples=1024*self.dim,
                        gen_candidates=gen_candidates_torch
                    ) 

                # print(candidates.shape)
                pessimistic_acq = pessimistic_CB.forward(self.x.unsqueeze(1))
                # print(f"UCB {UCB_acq} and LCB {torch.max(LCB_acq)}")
                
                kappa = maximize_factor*optimistic_acq - torch.max(maximize_factor*pessimistic_acq)
    
            
            # 7.1. Get the posterior mean for old and new GPs at the new and old best points.
            # new_config_x and old_config_x should be the configurations corresponding to the current
            # and previous best indices, respectively.
            x_pair = torch.stack([self.x[-1], self.x[-2]])

            # 7.2. Get posterior mean and covariance from the new model.
            new_posterior = self.model.posterior(x_pair)
            new_mean = new_posterior.mean         # Shape: [2]
            new_covar = new_posterior.mvn.covariance_matrix     # Shape: [2, 2]

            # 7.3. Get posterior mean and covariance from the old model.
            old_posterior = self.old_model.posterior(x_pair)
            old_mean = old_posterior.mean           # Shape: [2]
            old_covar = old_posterior.mvn.covariance_matrix       # Shape: [2, 2]

            # 7.4. Compute delta_mu (the absolute change in best posterior mean)
            # Here, we assume that new_config_x corresponds to the current best (new point)
            # and old_config_x corresponds to the previous best.
            delta_mu = abs(old_mean[1].item() - new_mean[0].item())
            
            # 7.6. Compute KL divergence between old and new posteriors at the new point.
            old_var = old_covar[0, 0].clamp(min=1e-12)
            new_var = new_covar[0, 0].clamp(min=1e-12)
            old_mu_val = old_mean[0]
            new_mu_val = new_mean[0]
            kl = 0.5 * (torch.log(new_var / old_var) +
                        (old_var + (old_mu_val - new_mu_val).pow(2)) / new_var - 1).item()

            # 7.7. Compute ei_diff, the expected-improvement gap difference.
            # If new_config_x and old_config_x are (approximately) equal, we set ei_diff to zero.
            if not torch.allclose(x_pair[0], x_pair[1], atol=1e-6):
                # We use the new model's posterior for these two points.
                # new_mean and new_covar already contain the predictions.
                # Compute the difference in means:
                g = (new_mean[0] - new_mean[1]).item()
                # Compute the effective variance difference
                diff_var = (new_covar[0, 0] - 2 * new_covar[0, 1] + new_covar[1, 1]).item()
                if diff_var < 0:
                    beta_val = 0.0
                    pdf_val = np.sqrt(1.0 / (2 * np.pi))
                    cdf_val = 1.0
                else:
                    beta_val = np.sqrt(diff_var)
                    u = g / beta_val if beta_val > 0 else 0.0
                    pdf_val = norm.pdf(u)
                    cdf_val = norm.cdf(u)
                ei_diff = beta_val * pdf_val + g * cdf_val
            else:
                ei_diff = 0.0

            # 7.8. Final expression for ΔR̃_t (the expected minimal regret gap).
            exp_min_regret_gap = delta_mu + ei_diff + kappa.item() * np.sqrt(0.5 * kl)
            self.stopping_history[key].append(exp_min_regret_gap.item())
            
    def update_cost(self, new_point):
        if callable(self.cost):
            # If self.cost is a function, call it and update cumulative cost
            new_cost = self.cost(new_point).view(-1)
            self.c = torch.cat((self.c, new_cost))
            self.cumulative_cost += new_cost.item()
        elif callable(self.objective_cost):
            new_value, new_cost = self.objective_cost(new_point)
            self.c = torch.cat((self.c, new_cost))
            self.cumulative_cost += new_cost.sum().item()
        else:
            # If self.cost is not a function, just increment cumulative cost by self.cost
            self.cumulative_cost += self.cost.item()

        self.cost_history.append(self.cumulative_cost)


    def print_iteration_info(self, iteration):
        print(f"Iteration {iteration}, New point: {self.x[-1].squeeze().detach().numpy()}, New value: {self.y[-1].detach().numpy()}")
        print("Best observed value:", self.best_f)
        print("Current acquisition value:", self.current_acq)
        print("Cumulative cost:", self.cumulative_cost)
        if hasattr(self, 'need_lmbda_update'):
            print("Gittins lmbda:", self.lmbda_history[-1])
        print("Running time:", self.runtime)
        print()

    def run(self, num_iterations, acquisition_function_class, **acqf_kwargs):
        self.budget = num_iterations
        if acquisition_function_class in (GittinsIndex, StableGittinsIndex):
            self.lmbda_history = []
            if acqf_kwargs.get('step_EIpu') == True:
                self.current_lmbda = None
                self.need_lmbda_update = True
            if acqf_kwargs.get('step_divide') == True:
                self.current_lmbda = acqf_kwargs['init_lmbda']
                self.need_lmbda_update = False
                                
        for i in range(num_iterations):
            start = time.process_time()
            self.iterate(acquisition_function_class, **acqf_kwargs)
            end = time.process_time()
            runtime = end - start
            self.runtime = runtime
            self.runtime_history.append(runtime)
            self.print_iteration_info(i)

    def get_best_value(self):
        return self.best_f


    def get_best_history(self):
        return self.best_history


    def get_cumulative_cost(self):
        return self.cumulative_cost


    def get_cost_history(self):
        return self.cost_history


    def get_regret_history(self, global_optimum):
        """
        Compute the regret history.

        Parameters:
        - global_optimum (float): The global optimum value of the objective function.

        Returns:
        - list: The regret history.
        """
        return [global_optimum - f if self.maximize else f - global_optimum for f in self.best_history]

    def get_lmbda_history(self):
        return self.lmbda_history

    def get_acq_history(self):
        return self.acq_history

    def get_runtime_history(self):
        return self.runtime_history
    
    def get_stopping_history(self):
        return self.stopping_history