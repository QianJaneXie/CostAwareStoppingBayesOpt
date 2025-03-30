import torch
import numpy as np
import math
from pandora_automl.utils import fit_gp_model
from pandora_automl.acquisition.stable_gittins import StableGittinsIndex
from botorch.acquisition import UpperConfidenceBound
from pandora_automl.acquisition.lcb import LowerConfidenceBound
from pandora_automl.acquisition.log_ei_puc import LogExpectedImprovementWithCost
import gc
import wandb

# Set default tensor type to float64
torch.set_default_dtype(torch.float64)

import os
import importlib.util

# Go up to the project root (PandoraAutoML)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Path to api.py
api_path = os.path.join(project_root, "LCBench", "api.py")

# Load the api module dynamically
spec = importlib.util.spec_from_file_location("lcb_api", api_path)
lcb_api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lcb_api)

# Use Benchmark from api.py
Benchmark = lcb_api.Benchmark

# Load the benchmark file
bench_path = os.path.join(project_root, "LCBench", "cached", "six_datasets_lw.json")

bench = Benchmark(bench_path, cache=False)

def normalize_config(config):
    # Convert each value to a torch tensor (ensuring float type for calculations)
    batch = torch.tensor(config["batch_size"])
    lr = torch.tensor(config["learning_rate"])
    units = torch.tensor(config["max_units"])
    momentum = torch.tensor(config["momentum"])
    weight_decay = torch.tensor(config["weight_decay"])
    layers = torch.tensor(float(config["num_layers"]))
    dropout = torch.tensor(config["max_dropout"])
    
    # For log-scaled parameters: batch size, learning rate, and max units.
    batch_norm = (torch.log(batch) - torch.log(torch.tensor(16.0))) / (torch.log(torch.tensor(512.0)) - torch.log(torch.tensor(16.0)))
    lr_norm = (torch.log(lr) - torch.log(torch.tensor(1e-4))) / (torch.log(torch.tensor(1e-1)) - torch.log(torch.tensor(1e-4)))
    units_norm = (torch.log(units) - torch.log(torch.tensor(64.0))) / (torch.log(torch.tensor(1024.0)) - torch.log(torch.tensor(64.0)))
    
    # For linearly scaled parameters.
    momentum_norm = (momentum - 0.1) / (0.99 - 0.1)
    weight_decay_norm = (weight_decay - 1e-5) / (1e-1 - 1e-5)
    layers_norm = (layers - 1) / (4 - 1)
    
    # Dropout is already between 0 and 1.
    dropout_norm = dropout

    # Combine into a 7-dimensional tensor.
    normalized_vector = torch.stack([
        batch_norm, 
        lr_norm, 
        momentum_norm, 
        weight_decay_norm, 
        layers_norm, 
        units_norm, 
        dropout_norm
    ])
    
    return normalized_vector

def run_bayesopt_experiment(bayesopt_config):
    print(bayesopt_config)
    dataset_name = bayesopt_config['dataset_name']
    seed = bayesopt_config['seed']
    output_standardize = bayesopt_config['output_standardize']
    maximize = bayesopt_config['maximize']
    dim = bayesopt_config['dim']
    n_iter = bayesopt_config['num_iteration']
    acq = bayesopt_config['acquisition_function']

    all_x = []
    all_y = []
    all_c = []
    for config_id in bench.data[dataset_name].keys():
        config = bench.query(dataset_name, "config", config_id)
        all_x.append(normalize_config(config))
        val_ce = bench.query(dataset_name, "final_val_cross_entropy", config_id)
        all_y.append(val_ce)
        runtime = bench.query(dataset_name, "time", config_id)[-1]
        all_c.append(runtime)

    all_x = torch.stack(all_x)
    all_y = torch.tensor(all_y).unsqueeze(1)
    all_c = torch.tensor(all_c).unsqueeze(1)

    torch.manual_seed(seed)
    init_config_id = torch.randint(low=0, high=2000, size=(2*(dim+1),))
    config_id_history = init_config_id.tolist()
    print(f"  Initial config id: {config_id_history}")
    x = all_x[init_config_id]
    y = all_y[init_config_id]
    c = all_c[init_config_id]
    acq_history = [np.nan]
    best_y_history = [y.min().item()]
    best_id_history = [config_id_history[y.argmin().item()]]
    cost_history = [0]
    StablePBGI_5e_6_acq_history = [np.nan]
    StablePBGI_1e_6_acq_history = [np.nan]
    StablePBGI_5e_7_acq_history = [np.nan]
    LogEIC_acq_history = [np.nan]
    regret_upper_bound_history = [np.nan]
    if acq == "StablePBGI-D":
        cur_lmbda = 0.001
        lmbda_history = [cur_lmbda]

    for i in range(n_iter):
        # 1. Fit a GP model on the current data.
        model = fit_gp_model(X=x, objective_X=y, cost_X=c, unknown_cost=True, output_standardize=output_standardize)
        
        # 2. Determine the best observed objective value.
        best_f = y.min()
            
        # 3. Define the acquisition function.
        StablePBGI_5e_6 = StableGittinsIndex(model=model, maximize=maximize, lmbda=5e-6, unknown_cost=True)
        StablePBGI_1e_6 = StableGittinsIndex(model=model, maximize=maximize, lmbda=1e-6, unknown_cost=True)
        StablePBGI_5e_7 = StableGittinsIndex(model=model, maximize=maximize, lmbda=5e-7, unknown_cost=True)
        LogEIC = LogExpectedImprovementWithCost(model=model, best_f=best_f, maximize=maximize, unknown_cost=True)
        single_outcome_model = fit_gp_model(X=x, objective_X=y, output_standardize=output_standardize)
        UCB = UpperConfidenceBound(model=single_outcome_model, maximize=maximize, beta=2 * np.log(dim * ((i + 1) ** 2) * (math.pi ** 2) / (6 * 0.1)) / 5)
        LCB = LowerConfidenceBound(model=single_outcome_model, maximize=maximize, beta=2 * np.log(dim * ((i + 1) ** 2) * (math.pi ** 2) / (6 * 0.1)) / 5)

        # 4. Evaluate the acquisition function on all candidate x's.
        # The unsqueeze operations add extra dimensions if required by your model.
        StablePBGI_5e_6_acq = StablePBGI_5e_6.forward(all_x.unsqueeze(1))
        StablePBGI_5e_6_acq[config_id_history] = y.squeeze(-1)
        StablePBGI_1e_6_acq = StablePBGI_1e_6.forward(all_x.unsqueeze(1))
        StablePBGI_1e_6_acq[config_id_history] = y.squeeze(-1)
        StablePBGI_5e_7_acq = StablePBGI_5e_7.forward(all_x.unsqueeze(1))
        StablePBGI_5e_7_acq[config_id_history] = y.squeeze(-1)
        LogEIC_acq = LogEIC.forward(all_x.unsqueeze(1))
        UCB_acq = UCB.forward(all_x.unsqueeze(1))
        LCB_acq = LCB.forward(all_x.unsqueeze(1))

        # 5. Record information for stopping.
        StablePBGI_5e_6_acq_history.append(torch.min(StablePBGI_5e_6_acq).item())
        StablePBGI_1e_6_acq_history.append(torch.min(StablePBGI_1e_6_acq).item())
        StablePBGI_5e_7_acq_history.append(torch.min(StablePBGI_5e_7_acq).item())
        LogEIC_acq_history.append(torch.max(LogEIC_acq).item())
        regret_upper_bound_history.append(torch.min(UCB_acq).item() - torch.min(LCB_acq).item())

        # 6. Select the candidate with the optimal acquisition value.
        num_configs = 2000
        all_ids = torch.arange(num_configs)
        mask = torch.ones(num_configs, dtype=torch.bool)
        mask[config_id_history] = False
        candidate_ids = all_ids[mask]
        if acq == "StablePBGI(5e-6)":
            candidate_acqs = StablePBGI_5e_6_acq[mask]
            new_config_id = candidate_ids[torch.argmin(candidate_acqs)]
            new_config_acq = torch.min(candidate_acqs)
        if acq == "StablePBGI(1e-6)":
            candidate_acqs = StablePBGI_1e_6_acq[mask]
            new_config_id = candidate_ids[torch.argmin(candidate_acqs)]
            new_config_acq = torch.min(candidate_acqs)
        if acq == "StablePBGI(5e-7)":
            candidate_acqs = StablePBGI_5e_7_acq[mask]
            new_config_id = candidate_ids[torch.argmin(candidate_acqs)]
            new_config_acq = torch.min(candidate_acqs)
        if acq == "LogEIC":
            candidate_acqs = LogEIC_acq[mask]
            new_config_id = candidate_ids[torch.argmax(candidate_acqs)]
            new_config_acq = torch.max(candidate_acqs)
        if acq == "LCB":
            candidate_acqs = LCB_acq[mask]
            new_config_id = candidate_ids[torch.argmin(candidate_acqs)]
            new_config_acq = torch.min(candidate_acqs)
        if acq == "StablePBGI-D":
            StablePBGI_D = StableGittinsIndex(model=model, maximize=maximize, lmbda=cur_lmbda, unknown_cost=True)
            StablePBGI_D_acq = StablePBGI_D.forward(all_x.unsqueeze(1))
            StablePBGI_D_acq[config_id_history] = y.squeeze(-1)
            new_config_id = torch.argmin(StablePBGI_D_acq)
            new_config_acq = torch.min(StablePBGI_D_acq)
            if new_config_acq >= best_f:
                cur_lmbda = cur_lmbda / 10
            lmbda_history.append(cur_lmbda)

        new_config_x = all_x[new_config_id]
        
        # 7. Query the objective for the new configuration.
        new_config_y = all_y[new_config_id]
        new_config_c = all_c[new_config_id]
        
        # 8. Append the new data to our training set.
        x = torch.cat([x, new_config_x.unsqueeze(0)], dim=0)
        y = torch.cat([y, new_config_y.unsqueeze(0)], dim=0)
        c = torch.cat([c, new_config_c.unsqueeze(0)], dim=0)
        config_id_history.append(new_config_id.item())
        acq_history.append(new_config_acq.item())
        best_y_history.append(best_f.item())
        best_id_history.append(config_id_history[y.argmin().item()])
        cost_history.append(new_config_c.item())

        print(f"Iteration {i + 1}:")
        print(f"  Selected config_id: {new_config_id}")
        print(f"  Acquisition value: {new_config_acq.item():.4f}")
        print(f"  Objective (final_val_cross_entropy): {new_config_y.item():.4f}")
        print(f"  Cost (time): {new_config_c.item():.4f}")
        print(f"  Current best observed: {best_f.item():.4f}")
        print()

        del model, single_outcome_model
        del StablePBGI_5e_6, StablePBGI_1e_6, StablePBGI_5e_7
        del LogEIC, UCB, LCB
        gc.collect()


    best_y_history.append(y.min().item())

    if acq == 'StablePBGI-D':
        return (cost_history,
                [best_id_history[0]]+config_id_history[-n_iter:],
                best_id_history, 
                best_y_history,
                acq_history, 
                StablePBGI_5e_6_acq_history,
                StablePBGI_1e_6_acq_history,
                StablePBGI_5e_7_acq_history,  
                LogEIC_acq_history,
                regret_upper_bound_history,
                lmbda_history)
    else:
        return (cost_history,
                [best_id_history[0]]+config_id_history[-n_iter:], 
                best_id_history,
                best_y_history,
                acq_history,
                StablePBGI_5e_6_acq_history,
                StablePBGI_1e_6_acq_history, 
                StablePBGI_5e_7_acq_history, 
                LogEIC_acq_history,
                regret_upper_bound_history)

wandb.init()

result = run_bayesopt_experiment(wandb.config)

if wandb.config["acquisition_function"] == "StablePBGI-D":
    (cost_history, config_id_history, best_id_history, best_y_history, acq_history,
     StablePBGI_5e_6_acq_history, StablePBGI_1e_6_acq_history,
     StablePBGI_5e_7_acq_history, LogEIC_acq_history,
     regret_upper_bound_history, lmbda_history) = result
else:
    (cost_history, config_id_history, best_id_history, best_y_history, acq_history,
     StablePBGI_5e_6_acq_history, StablePBGI_1e_6_acq_history,
     StablePBGI_5e_7_acq_history, LogEIC_acq_history,
     regret_upper_bound_history) = result

cumulative_costs = np.cumsum(cost_history)

# Log full info
for idx in range(len(cost_history)):
    log_dict = {
        "config id": config_id_history[idx],
        "cumulative cost": cumulative_costs[idx],
        "current best id": best_id_history[idx],
        "current best observed": best_y_history[idx],
        "acq": acq_history[idx],
        "StablePBGI(5e-6) acq": StablePBGI_5e_6_acq_history[idx],
        "StablePBGI(1e-6) acq": StablePBGI_1e_6_acq_history[idx],
        "StablePBGI(5e-7) acq": StablePBGI_5e_7_acq_history[idx],
        "LogEIC acq": LogEIC_acq_history[idx],
        "regret upper bound": regret_upper_bound_history[idx],
    }
    if wandb.config["acquisition_function"] == "StablePBGI-D":
        log_dict["lmbda"] = lmbda_history[idx]

    wandb.log(log_dict)

wandb.finish()