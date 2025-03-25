import torch
import numpy as np
import math
from pandora_automl.utils import fit_gp_model
from pandora_automl.acquisition.gittins import GittinsIndex
from botorch.acquisition import LogExpectedImprovement, UpperConfidenceBound
from pandora_automl.acquisition.lcb import LowerConfidenceBound
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
    for config_id in bench.data[dataset_name].keys():
        config = bench.query(dataset_name, "config", config_id)
        all_x.append(normalize_config(config))
        val_ce = bench.query(dataset_name, "final_val_cross_entropy", config_id)
        all_y.append(val_ce)

    all_x = torch.stack(all_x)
    all_y = torch.tensor(all_y).unsqueeze(1)

    torch.manual_seed(seed)
    init_config_id = torch.randint(low=0, high=2000, size=(1,)).item()
    print(f"  Initial config id: {init_config_id}")
    config_id_history = [init_config_id]
    x = all_x[init_config_id].unsqueeze(0)
    y = all_y[init_config_id].unsqueeze(0)
    acq_history = [np.nan]
    current_best_history = []
    PBGI_5e_3_acq_history = [np.nan]
    PBGI_1e_3_acq_history = [np.nan]
    PBGI_5e_4_acq_history = [np.nan]
    LogEI_acq_history = [np.nan]
    regret_upper_bound_history = [np.nan]
    if acq == "PBGI-D":
        cur_lmbda = 0.1
        lmbda_history = [cur_lmbda]

    for i in range(n_iter):
        # 1. Fit a GP model on the current data.
        model = fit_gp_model(X=x, objective_X=y, output_standardize=output_standardize)
        
        # 2. Determine the best observed objective value.
        best_f = y.min()
        
        # 3. Define the acquisition function.
        PBGI_5e_3 = GittinsIndex(model=model, maximize=maximize, lmbda=5e-3)
        PBGI_1e_3 = GittinsIndex(model=model, maximize=maximize, lmbda=1e-3)
        PBGI_5e_4 = GittinsIndex(model=model, maximize=maximize, lmbda=5e-4)
        LogEI = LogExpectedImprovement(model=model, best_f=best_f, maximize=maximize)
        single_outcome_model = fit_gp_model(X=x, objective_X=y, output_standardize=output_standardize)
        UCB = UpperConfidenceBound(model=single_outcome_model, maximize=maximize, beta=2 * np.log(dim * ((i + 1) ** 2) * (math.pi ** 2) / (6 * 0.1)) / 5)
        LCB = LowerConfidenceBound(model=single_outcome_model, maximize=maximize, beta=2 * np.log(dim * ((i + 1) ** 2) * (math.pi ** 2) / (6 * 0.1)) / 5)

        # 4. Evaluate the acquisition function on all candidate x's.
        # The unsqueeze operations add extra dimensions if required by your model.
        PBGI_5e_3_acq = PBGI_5e_3.forward(all_x.unsqueeze(1))
        PBGI_1e_3_acq = PBGI_1e_3.forward(all_x.unsqueeze(1))
        PBGI_5e_4_acq = PBGI_5e_4.forward(all_x.unsqueeze(1))
        LogEI_acq = LogEI.forward(all_x.unsqueeze(1))
        UCB_acq = UCB.forward(all_x.unsqueeze(1))
        LCB_acq = LCB.forward(all_x.unsqueeze(1))

        # 5. Record information for stopping.
        PBGI_5e_3_acq_history.append(torch.min(PBGI_5e_3_acq).item())
        PBGI_1e_3_acq_history.append(torch.min(PBGI_1e_3_acq).item())
        PBGI_5e_4_acq_history.append(torch.min(PBGI_5e_4_acq).item())
        LogEI_acq_history.append(torch.max(LogEI_acq).item())
        regret_upper_bound_history.append(torch.min(UCB_acq).item() - torch.min(LCB_acq).item())

        # 6. Select the candidate with the lowest acquisition value.
        if acq == "PBGI(5e-3)":
            new_config_id = torch.argmin(PBGI_5e_3_acq)
            new_config_acq = torch.min(PBGI_5e_3_acq)
        if acq == "PBGI(1e-3)":
            new_config_id = torch.argmin(PBGI_1e_3_acq)
            new_config_acq = torch.min(PBGI_1e_3_acq)
        if acq == "PBGI(5e-4)":
            new_config_id = torch.argmin(PBGI_5e_4_acq)
            new_config_acq = torch.min(PBGI_5e_4_acq)
        if acq == "LogEI":
            new_config_id = torch.argmax(LogEI_acq)
            new_config_acq = torch.max(LogEI_acq)
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
        
        # 8. Append the new data to our training set.
        x = torch.cat([x, new_config_x.unsqueeze(0)], dim=0)
        y = torch.cat([y, new_config_y.unsqueeze(0)], dim=0)
        config_id_history.append(new_config_id.item())
        acq_history.append(new_config_acq.item())
        current_best_history.append(best_f.item())

    current_best_history.append(y.min().item())

    if bayesopt_config["acq"] == 'PBGI-D':
        return (config_id_history, 
                current_best_history,
                acq_history, 
                PBGI_5e_3_acq_history, 
                PBGI_1e_3_acq_history,
                PBGI_5e_4_acq_history, 
                LogEI_acq_history,
                regret_upper_bound_history,
                lmbda_history)
    else:
        return (config_id_history, 
                current_best_history,
                acq_history,
                PBGI_5e_3_acq_history, 
                PBGI_1e_3_acq_history,
                PBGI_5e_4_acq_history, 
                LogEI_acq_history,
                regret_upper_bound_history)

wandb.init()
if wandb.config["acquisition_function"] == "PBGI-D":
    (config_id_history, current_best_history, acq_history, PBGI_5e_3_acq_history, PBGI_1e_3_acq_history, PBGI_5e_4_acq_history, LogEI_acq_history, regret_upper_bound_history, lmbda_history) = run_bayesopt_experiment(wandb.config)
    for iter, config_id, current_best, acq, PBGI_5e_3_acq, PBGI_1e_3_acq, PBGI_5e_4_acq, LogEI_acq, regret_upper_bound, lmbda in zip(range(wandb.config["num_iteration"]), config_id_history, current_best_history, PBGI_5e_3_acq_history, PBGI_1e_3_acq_history, LogEI_acq_history, regret_upper_bound_history, lmbda_history):
        wandb.log({"iteration": iter, "config id": config_id, "current best observed": current_best, "acq": acq, "PBGI(5e-3) acq": PBGI_5e_3_acq, "PBGI(1e-3) acq": PBGI_1e_3_acq, "PBGI(5e-4) acq": PBGI_5e_4_acq, "LogEI acq": LogEI_acq, "regret upper bound": regret_upper_bound, "lmbda": lmbda})
else:
    (config_id_history, current_best_history, acq_history, PBGI_5e_3_acq_history, PBGI_1e_3_acq_history, PBGI_5e_4_acq_history, LogEI_acq_history, regret_upper_bound_history) = run_bayesopt_experiment(wandb.config)
    for iter, config_id, current_best, acq, PBGI_5e_3_acq, PBGI_1e_3_acq, PBGI_5e_4_acq, LogEI_acq, regret_upper_bound in zip(range(wandb.config["num_iteration"]), config_id_history, current_best_history, PBGI_5e_3_acq_history, PBGI_1e_3_acq_history, LogEI_acq_history, regret_upper_bound_history):
        wandb.log({"iteration": iter, "config id": config_id, "current best observed": current_best, "acq": acq, "PBGI(5e-3) acq": PBGI_5e_3_acq, "PBGI(1e-3) acq": PBGI_1e_3_acq, "PBGI(5e-4) acq": PBGI_5e_4_acq, "LogEI acq": LogEI_acq, "regret upper bound": regret_upper_bound})
wandb.finish()