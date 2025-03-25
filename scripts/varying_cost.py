import torch
import numpy as np
import math
from pandora_automl.utils import fit_gp_model
from pandora_automl.acquisition.gittins import GittinsIndex
from botorch.acquisition import UpperConfidenceBound
from pandora_automl.acquisition.lcb import LowerConfidenceBound
from pandora_automl.acquisition.log_ei_puc import LogExpectedImprovementWithCost
import wandb

# use a GPU if available
torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    init_config_id = torch.randint(low=0, high=2000, size=(1,)).item()
    print(f"  Initial config id: {init_config_id}")
    config_id_history = [init_config_id]
    x = all_x[init_config_id].unsqueeze(0)
    y = all_y[init_config_id].unsqueeze(0)
    c = all_c[init_config_id].unsqueeze(0)
    acq_history = [np.nan]
    current_best_history = []
    cost_history = [0]
    PBGI_1e_5_acq_history = [np.nan]
    PBGI_5e_6_acq_history = [np.nan]
    PBGI_1e_6_acq_history = [np.nan]
    LogEIC_acq_history = [np.nan]
    regret_upper_bound_history = [np.nan]
    if acq == "PBGI-D":
        cur_lmbda = 0.001
        lmbda_history = [cur_lmbda]

    for i in range(n_iter):
        # 1. Fit a GP model on the current data.
        model = fit_gp_model(X=x, objective_X=y, cost_X=c, unknown_cost=True, output_standardize=output_standardize)
        
        # 2. Determine the best observed objective value.
        best_f = y.min()
        
        # 3. Define the acquisition function.
        PBGI_1e_5 = GittinsIndex(model=model, maximize=maximize, lmbda=1e-5, unknown_cost=True)
        PBGI_5e_6 = GittinsIndex(model=model, maximize=maximize, lmbda=5e-6, unknown_cost=True)
        PBGI_1e_6 = GittinsIndex(model=model, maximize=maximize, lmbda=1e-6, unknown_cost=True)
        LogEIC = LogExpectedImprovementWithCost(model=model, best_f=best_f, maximize=maximize, unknown_cost=True)
        single_outcome_model = fit_gp_model(X=x, objective_X=y, output_standardize=output_standardize)
        UCB = UpperConfidenceBound(model=single_outcome_model, maximize=maximize, beta=2 * np.log(dim * ((i + 1) ** 2) * (math.pi ** 2) / (6 * 0.1)) / 5)
        LCB = LowerConfidenceBound(model=single_outcome_model, maximize=maximize, beta=2 * np.log(dim * ((i + 1) ** 2) * (math.pi ** 2) / (6 * 0.1)) / 5)

        # 4. Evaluate the acquisition function on all candidate x's.
        # The unsqueeze operations add extra dimensions if required by your model.
        PBGI_1e_5_acq = PBGI_1e_5.forward(all_x.unsqueeze(1))
        PBGI_5e_6_acq = PBGI_5e_6.forward(all_x.unsqueeze(1))
        PBGI_1e_6_acq = PBGI_1e_6.forward(all_x.unsqueeze(1))
        LogEIC_acq = LogEIC.forward(all_x.unsqueeze(1))
        UCB_acq = UCB.forward(all_x.unsqueeze(1))
        LCB_acq = LCB.forward(all_x.unsqueeze(1))

        # 5. Record information for stopping.
        PBGI_1e_5_acq_history.append(torch.min(PBGI_1e_5_acq).item())
        PBGI_5e_6_acq_history.append(torch.min(PBGI_5e_6_acq).item())
        PBGI_1e_6_acq_history.append(torch.min(PBGI_1e_6_acq).item())
        LogEIC_acq_history.append(torch.max(LogEIC_acq).item())
        regret_upper_bound_history.append(torch.min(UCB_acq).item() - torch.min(LCB_acq).item())

        # 6. Select the candidate with the lowest acquisition value.
        if acq == "PBGI(1e-5)":
            new_config_id = torch.argmin(PBGI_1e_5_acq)
            new_config_acq = torch.min(PBGI_1e_5_acq)
        if acq == "PBGI(5e-6)":
            new_config_id = torch.argmin(PBGI_5e_6_acq)
            new_config_acq = torch.min(PBGI_5e_6_acq)
        if acq == "PBGI(1e-6)":
            new_config_id = torch.argmin(PBGI_1e_6_acq)
            new_config_acq = torch.min(PBGI_1e_6_acq)
        if acq == "LogEIC":
            new_config_id = torch.argmax(LogEIC_acq)
            new_config_acq = torch.max(LogEIC_acq)
        if acq == "LCB":
            new_config_id = torch.argmin(LCB_acq)
            new_config_acq = torch.min(LCB_acq)
        if acq == "PBGI-D":
            PBGI_D = GittinsIndex(model=model, maximize=maximize, lmbda=cur_lmbda, unknown_cost=True)
            PBGI_D_acq = PBGI_D.forward(all_x.unsqueeze(1))
            new_config_id = torch.argmin(PBGI_D_acq)
            new_config_acq = torch.min(PBGI_D_acq)
            if new_config_acq > best_f:
                cur_lmbda = cur_lmbda / 2
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
        current_best_history.append(best_f.item())
        cost_history.append(new_config_c.item())

    current_best_history.append(y.min().item())

    if acq == 'PBGI-D':
        return (config_id_history, 
                current_best_history,
                acq_history, 
                PBGI_1e_5_acq_history, 
                PBGI_5e_6_acq_history,
                PBGI_1e_6_acq_history, 
                LogEIC_acq_history,
                regret_upper_bound_history,
                lmbda_history)
    else:
        return (config_id_history, 
                current_best_history,
                acq_history,
                PBGI_1e_5_acq_history, 
                PBGI_5e_6_acq_history,
                PBGI_1e_6_acq_history, 
                LogEIC_acq_history,
                regret_upper_bound_history)

wandb.init()
if wandb.config["acquisition_function"] == "PBGI-D":
    (cost_history, config_id_history, current_best_history, acq_history, PBGI_1e_5_acq_history, PBGI_5e_6_acq_history, PBGI_1e_6_acq_history, LogEIC_acq_history, regret_upper_bound_history, lmbda_history) = run_bayesopt_experiment(wandb.config)
    for cum_cost, config_id, current_best, acq, PBGI_1e_5_acq, PBGI_5e_6_acq, PBGI_1e_6_acq, LogEIC_acq, regret_upper_bound, lmbda in zip(np.cumsum(cost_history), config_id_history, current_best_history, PBGI_1e_5_acq_history, PBGI_5e_6_acq_history, PBGI_1e_6_acq_history, LogEIC_acq_history, regret_upper_bound_history, lmbda_history):
        wandb.log({"cumulative cost": cum_cost, "config id": config_id, "current best observed": current_best, "acq": acq, "PBGI(1e-5) acq": PBGI_1e_5_acq, "PBGI(5e-6) acq": PBGI_5e_6_acq, "PBGI(1e-6) acq": PBGI_1e_6_acq, "LogEIC acq": LogEIC_acq, "regret upper bound": regret_upper_bound, "lmbda": lmbda})
else:
    (cost_history, config_id_history, current_best_history, acq_history, PBGI_1e_5_acq_history, PBGI_5e_6_acq_history, PBGI_1e_6_acq_history, LogEIC_acq_history, regret_upper_bound_history) = run_bayesopt_experiment(wandb.config)
    for cum_cost, config_id, current_best, acq, PBGI_1e_5_acq, PBGI_5e_6_acq, PBGI_1e_6_acq, LogEIC_acq, regret_upper_bound in zip(np.cumsum(cost_history), config_id_history, current_best_history, PBGI_1e_5_acq_history, PBGI_5e_6_acq_history, PBGI_1e_6_acq_history, LogEIC_acq_history, regret_upper_bound_history):
        wandb.log({"cumulative cost": cum_cost, "config id": config_id, "current best observed": current_best, "acq": acq, "PBGI(1e-5) acq": PBGI_1e_5_acq, "PBGI(5e-6) acq": PBGI_5e_6_acq, "PBGI(1e-6) acq": PBGI_1e_6_acq, "LogEIC acq": LogEIC_acq, "regret upper bound": regret_upper_bound})
wandb.finish()