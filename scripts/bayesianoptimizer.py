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

        is_rs = False
        is_ms = False
        is_ts = False
        is_pes = False

        if acquisition_function_class == BudgetedMultiStepLookaheadEI:
            gaussian_likelihood = True
        else:
            gaussian_likelihood = False
        
        if acquisition_function_class == "RandomSearch":
            is_rs = True
            new_point = torch.rand(1, self.dim)
            
        else:
            if acquisition_function_class in (ExpectedImprovementWithCost, LogExpectedImprovementWithCost, GittinsIndex, BudgetedMultiStepLookaheadEI):
                unknown_cost = self.unknown_cost
            else:
                unknown_cost = False

            # print("time_1", time.time())
            self.old_model = self.model
            model = fit_gp_model(
                X=self.x.detach(), 
                objective_X=self.y.detach(), 
                cost_X=self.c.detach(), 
                unknown_cost=unknown_cost,
                kernel=self.kernel,
                gaussian_likelihood=gaussian_likelihood,
                # noisy_observation=self.noisy_observation,
                output_standardize=self.output_standardize,
            )
            self.model = model
            if (self.old_model is None):
                self.old_model = model

            # print("time_2", time.time())

            acqf_args = {'model': model}
           
            if acquisition_function_class in ("ThompsonSampling", qPredictiveEntropySearch, "SurrogatePrice"):
            
                # Draw sample path(s)
                paths = draw_matheron_paths(model, sample_shape=torch.Size([1]))
                
                # Optimize
                optimal_input, optimal_output = optimize_posterior_samples(paths=paths, bounds=self.bounds, maximize=self.maximize)

                if acquisition_function_class == "ThompsonSampling":
                    is_ts = True
                    new_point = optimal_input
                    self.current_acq = optimal_output.item()
                
                elif acquisition_function_class == qPredictiveEntropySearch:
                    is_pes = True
                    PES = qPredictiveEntropySearch(model=model, optimal_inputs=optimal_input, maximize=self.maximize)
                    new_point, new_point_PES = optimize_acqf(
                        acq_function=PES,
                        bounds=self.bounds,
                        q=1,
                        num_restarts=10*self.dim,
                        raw_samples=200*self.dim,
                        options={
                                "batch_limit": 5,
                                "maxiter": 200,
                                "with_grad": False
                            },
                    )
                    self.current_acq = new_point_PES.item()

            
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


            elif acquisition_function_class in (qMaxValueEntropy, qMultiFidelityMaxValueEntropy):
                candidate_set = torch.rand(1000*self.dim, self.bounds.size(1))
                candidate_set = self.bounds[0] + (self.bounds[1] - self.bounds[0]) * candidate_set
                acqf_args['candidate_set'] = candidate_set

                if acquisition_function_class == qMultiFidelityMaxValueEntropy:
                    cost_function = copy(self.cost)
                    class CostModel(GenericDeterministicModel):
                        def __init__(self):
                            super().__init__(f=cost_function)
                    cost_model = CostModel()
                    cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)
                    acqf_args['cost_aware_utility'] = cost_aware_utility


            elif acquisition_function_class in (MultiStepLookaheadEI, BudgetedMultiStepLookaheadEI):
                is_ms = True
                acqf_args['batch_size'] = 1
                acqf_args['lookahead_batch_sizes'] = [1, 1, 1]
                acqf_args['num_fantasies'] = [1, 1, 1]
                
                if acquisition_function_class == BudgetedMultiStepLookaheadEI:
                    acqf_args['cost_function'] = copy(self.cost)
                    acqf_args['unknown_cost'] = self.unknown_cost
                    acqf_args['budget_plus_cumulative_cost'] = min(self.budget - self.cumulative_cost, self.c[-4:].sum().item()) + self.c.sum().item()
                    print(acqf_args['budget_plus_cumulative_cost'])
                
            
            else:
                acqf_args.update(**acqf_kwargs)


            if is_ts == False and is_pes == False:
                acq_function = acquisition_function_class(**acqf_args)
                if self.suggested_x_full_tree is not None:
                    print("Using warmstart for multi-step initialization")

                    batch_initial_conditions = warmstart_multistep(
                            acq_function=acq_function,
                            bounds=self.bounds,
                            num_restarts=10 * self.dim,
                            raw_samples=200 * self.dim,
                            full_optimizer=self.suggested_x_full_tree,
                            algo_params=acqf_args,
                        )
                else:
                    batch_initial_conditions = None
                    '''
                    print("Using random batch initialization") 
                    batch_initial_conditions = gen_batch_initial_conditions(
                        acq_function=acq_function,
                        bounds=self.bounds,
                        q=1,
                        num_restarts=10 * self.dim,
                        raw_samples=200 * self.dim
                    )
                    '''
                q = acq_function.get_augmented_q_batch_size(1) if is_ms else 1

                # use grid search if dimension is 1, otherwise use optimization
                if self.dim == 1:
                    print("Using grid search for 1D optimization")
                    candidates = torch.linspace(0, 1, self.granularity).unsqueeze(1).unsqueeze(1)
                    # grids = [torch.linspace(0, 1, granularity) for _ in range(self.dim)]
                    # mesh = torch.meshgrid(*grids, indexing="ij")  # consistent indexing
                    # candidates = torch.stack([g.flatten() for g in mesh], dim=-1).unsqueeze(-2)  # shape (granularity^dim, dim)
                    candidates_acq_vals = acq_function.forward(candidates[self.mask])
                    candidates =  candidates.detach()
                    best_idx = torch.argmax(candidates_acq_vals.view(-1), dim=0)
                    best_point = candidates[best_idx]
                    best_acq_val = candidates_acq_vals[best_idx].item()

                else:
                    print("time_3", time.time())
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
                    '''
                    
                    candidates, candidates_acq_vals = optimize_acqf(
                        acq_function=acq_function,
                        bounds=self.bounds,
                        q=q,
                        num_restarts=10 * self.dim,
                        raw_samples=200 * self.dim,
                        options={
                                "batch_limit": 5,
                                "maxiter": 200,
                                "method": "L-BFGS-B",
                            },
                        batch_initial_conditions=batch_initial_conditions,
                        return_best_only=False,
                        return_full_tree=is_ms,
                    )
                    print("candidate")
                    print(candidates, candidates_acq_vals)
                    print("time_4", time.time())
                    best_idx = torch.argmax(candidates_acq_vals.view(-1), dim=0)
                    best_point = candidates[best_idx]
                    best_acq_val = candidates_acq_vals[best_idx].item()
                    '''
                
                # print("candidates:", candidates.shape)
            
                if is_ms:
                    # save all tree variables for multi-step initialization
                    self.suggested_x_full_tree = candidates.clone()
                    candidate = acq_function.extract_candidates(candidates)
                # best_point = candidate
                # best_acq_val = candidate_acq_val

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
        # self.log_time(self.update_stopping_criteria, "PRB")
        # self.log_time(self.update_stopping_criteria, "PRB", option=2)
        # self.log_time(self.update_stopping_criteria, "PRB", option=3)
        self.log_time(self.update_stopping_criteria, "StablePBGI")
        self.log_time(self.update_stopping_criteria, "LogEIC")
        self.log_time(self.update_stopping_criteria, "UCB-LCB")

        self.update_best()
        self.update_cost(new_point)
        print("New point:", new_point.detach())
        if (self.dim == 1):
            self.mask[int(new_point.detach()*(self.granularity - 1))] = False
        self.iteration += 1


        if is_rs:
            self.current_acq = new_value.item()

        self.acq_history.append(self.current_acq)

        # Check if lmbda needs to be updated in the next iteration
        if acquisition_function_class in (GittinsIndex, StableGittinsIndex) and (acqf_kwargs.get('step_EIpu') == True or acqf_kwargs.get('step_divide') == True):
            if (self.maximize and self.current_acq < self.best_f) or (not self.maximize and -self.current_acq > self.best_f):
                self.need_lmbda_update = True


    def log_time(self, func, *args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start_time:.4f} seconds")
        return result
    
    def update_stopping_criteria(self, stopping_criteria):
        '''
        This function implements the following stopping rules: 
                                'StablePBGI(1e-1)': [np.nan],
                                'StablePBGI(1e-2)': [np.nan],
                                'StablePBGI(1e-3)': [np.nan],
                                'LogEIC': [np.nan],
                                'regret upper bound': [np.nan],
                                'exp min regret gap': [np.nan],
                                'PRB': [np.nan]
        '''
        # Currently only works for dim=1
        # Initialization for probabilistic regret bound (PRB) stopping rule
        epsilon = 0.1
        # epsilons = [0.001, 0.01, 0.1]
        lmbdas = [0.1, 0.01, 0.001]
        
        candidates = torch.linspace(0, 1, self.granularity).unsqueeze(1)

        if (stopping_criteria == "PRB"):
            # Probabilistic regret bound
            key = f'PRB_{epsilon}'
            paths = draw_matheron_paths(self.model, sample_shape=torch.Size([self.num_samples]))
            bounds = torch.stack([torch.zeros(self.dim), torch.ones(self.dim)])
            
            # When to not skip PRB calculation
            # 1. If the iteration is less than 50 and the iteration is a multiple of 5
            # 2. If the iteration is a multiple of 10
            if (self.dim > 1):
                skip_PRB = not ((self.iteration < 50 and self.iteration % 5 == 0) or self.iteration % 10 == 0)
                if skip_PRB:
                    print("Skipping PRB calculation")
                    if key not in self.stopping_history:
                        self.stopping_history[key] = [np.nan]  # initialize if missing
                    self.stopping_history[key].append(self.stopping_history[key][-1])
                    return

            if (self.dim == 1):
                regrets = paths(candidates).max(dim=1).values - paths(self.best_x.unsqueeze(0)).squeeze(-1)
            else:
                # 1. build a QMC sampler that will internally draw your fantasy paths
                _, optimum_values = optimize_posterior_samples(paths=paths, 
                                                               bounds=bounds, 
                                                               raw_samples=200*self.dim, 
                                                               num_restarts=10*self.dim,
                                                               maximize=self.maximize)
                '''
                elif option == 2:
                    _, optimum_values = optimize_posterior_samples(paths=paths, bounds=bounds, raw_samples=1024, num_restarts=20, maximize=self.maximize) 
                elif option == 3:
                    _, optimum_values = optimize_posterior_samples(paths=paths, bounds=bounds, raw_samples=256*self.dim, num_restarts=10*self.dim, maximize=self.maximize)
                '''
                # print("optimum_values:", optimum_values)
                regrets = optimum_values.squeeze(-1) - paths(self.best_x.unsqueeze(0)).squeeze(-1)
    
                # print("regrets:", regrets)
            
            prb_estimate = (regrets <= epsilon).float().mean().item()
            if key not in self.stopping_history:
                self.stopping_history[key] = [np.nan]  # initialize if missing
            self.stopping_history[key].append(prb_estimate)
            
                # print("Probabilistic regret bound")
                # print(f'Epsilon: {epsilon}, regrets: {prb_estimate}, num_samples: {self.num_samples}')
            self.num_samples = min(math.ceil(self.num_samples * 1.5), 1000)
 
        elif (stopping_criteria == "StablePBGI"):
            # 3. Stable PBGI
            for lmbda in lmbdas:
                key = f'StablePBGI({lmbda})'
                if key not in self.stopping_history:
                    self.stopping_history[key] = [np.nan]  # initialize if missing
                StablePBGI = StableGittinsIndex(model=self.model, maximize=self.maximize, lmbda=lmbda, cost=self.cost)
                if (self.dim == 1): 
                    StablePBGI_acq = StablePBGI.forward(candidates.unsqueeze(1))
                    # TODO: change to minimize if self.maximize is False
                    new_config_acq = torch.max(StablePBGI_acq[self.mask]) 
                   
                else:
                    # Optimize the acquisition functio
                    candidates, StablePBGI_acq = optimize_acqf(
                        acq_function=StablePBGI,
                        bounds=self.bounds,
                        q=1,
                        num_restarts=10*self.dim,
                        raw_samples=1024*self.dim,
                        gen_candidates=gen_candidates_torch
                    )
     
                    new_config_acq = StablePBGI_acq # torch.max(StablePBGI_acq)

                self.stopping_history[key].append(new_config_acq.item())
                # print("StablePBGI")
                # print(f'Lambda: {lmbda}, acquisition: {new_config_acq.item()}') 

        elif (stopping_criteria == "LogEIC"):
            key = 'LogEIC'
            # key_1 = 'LogEIC_small'
            # key_2 = 'LogEIC_large'
            # for key in [key_1, key_2]:
            if key not in self.stopping_history:
                self.stopping_history[key] = [np.nan]  # initialize if missing
            LogEIC = LogExpectedImprovementWithCost(model=self.model, best_f=self.best_f, maximize=self.maximize, cost=self.cost)
            if (self.dim == 1):
                LogEIC_acq = LogEIC.forward(candidates.unsqueeze(1)) 
                new_config_acq = torch.max(LogEIC_acq[self.mask])
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
            key_1 = f'Expected-Min-Regret-Gap'
            key_2 = f'UCB-LCB'
            for key in [key_1, key_2]:
                if key not in self.stopping_history:
                    self.stopping_history[key] = [np.nan]  # initialize if missing 
            
            UCB = UpperConfidenceBound(model=self.model, maximize=self.maximize, beta=2 * np.log(self.dim * ((self.iteration + 1) ** 2) * (math.pi ** 2) / (6 * 0.1)) / 5)
            LCB = LowerConfidenceBound(model=self.model, maximize=self.maximize, beta=2 * np.log(self.dim * ((self.iteration + 1) ** 2) * (math.pi ** 2) / (6 * 0.1)) / 5)
            beta = 2 * np.log(self.dim * ((self.iteration + 1) ** 2) * (math.pi ** 2) / (6 * 0.1)) / 5 
            print(f"beta: {beta}")
            if (self.dim == 1):
                UCB_acq = UCB.forward(candidates.unsqueeze(1))
                LCB_acq = LCB.forward(self.x.unsqueeze(1))
                # LCB_acq = LCB.forward(candidates.unsqueeze(1))
                # print("mask:", self.mask)
                # 7.5. Compute κ_{t−1} = UCB - LCB gap. #[~self.mask]
                # print("mask", self.mask)
                # print("no mask", ~self.mask)
                # print(LCB_acq[~self.mask]) 
                kappa = torch.max(UCB_acq) - torch.max(LCB_acq)
            else:
                candidates, UCB_acq = optimize_acqf(
                        acq_function=UCB,
                        bounds=self.bounds,
                        q=1,
                        num_restarts=10*self.dim,
                        raw_samples=1024*self.dim,
                        gen_candidates=gen_candidates_torch
                    ) 

                print(candidates.shape)
                LCB_acq = LCB.forward(self.x.unsqueeze(1))
                print(f"UCB {UCB_acq} and LCB {torch.max(LCB_acq)}")
                kappa = UCB_acq - torch.max(LCB_acq)
                print(f"kappa: {kappa}")
                # kappa = torch.max(UCB_acq) - torch.max(LCB_acq)
            
            
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
            
            self.stopping_history[key_1].append(exp_min_regret_gap.item())
            
            self.stopping_history[key_2].append(kappa.item())

        

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
        # print("Stopping criteria history:", self.stopping_history)
        print()

    def run(self, num_iterations, acquisition_function_class, **acqf_kwargs):
        self.budget = num_iterations
        if acquisition_function_class in (GittinsIndex, StableGittinsIndex):
            # print("Gittins lmbda:", acqf_kwargs['lmbda'])
            # print("step_EIpu: ", acqf_kwargs.get('step_EIpu'))
            # print("step divide: ", acqf_kwargs.get('step_divide'))
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


    def run_until_budget(self, budget, acquisition_function_class, **acqf_kwargs):
        self.budget = budget
        if acquisition_function_class in (GittinsIndex, StableGittinsIndex):
            self.lmbda_history = []
            if acqf_kwargs.get('step_EIpu') == True:
                self.current_lmbda = None
                self.need_lmbda_update = True
            if acqf_kwargs.get('step_divide') == True:
                self.current_lmbda = acqf_kwargs['init_lmbda']
                self.need_lmbda_update = False

        i = 0
        while self.cumulative_cost < self.budget:
            start = time.process_time()
            self.iterate(acquisition_function_class, **acqf_kwargs)
            end = time.process_time()
            runtime = end - start
            self.runtime = runtime
            self.runtime_history.append(runtime)
            self.print_iteration_info(i)
            i += 1


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