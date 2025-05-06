import torch
import numpy as np
import math
from scipy.stats import norm
from pandora_automl.utils import fit_gp_model
from pandora_automl.acquisition.stable_gittins import StableGittinsIndex
from botorch.acquisition import UpperConfidenceBound
from pandora_automl.acquisition.lcb import LowerConfidenceBound
from pandora_automl.acquisition.log_ei_puc import LogExpectedImprovementWithCost
from botorch.sampling.pathwise import draw_matheron_paths
import gc
import wandb
import time

# Set default tensor type to float64
torch.set_default_dtype(torch.float64)


def run_bayesopt_experiment(bayesopt_config):
    print(bayesopt_config)
    seed = bayesopt_config['seed']
    output_standardize = bayesopt_config['output_standardize']
    maximize = bayesopt_config['maximize']
    dim = bayesopt_config['dim']
    n_iter = bayesopt_config['num_iteration']
    num_configs = bayesopt_config['num_configs']
    acq = bayesopt_config['acquisition_function']
    num_batches = 32
    batch_size = num_configs // num_batches

    # Load benchmark data
    nashpo_data = torch.load('scripts/nashpo_data.pt')
    all_x = nashpo_data['x']
    all_y = nashpo_data['y']
    all_c = nashpo_data['c']
    estimated_costs = nashpo_data['estimated_costs']

    # Sample initial configurations
    torch.manual_seed(seed)
    init_config_id = torch.randint(low=0, high=num_configs, size=(2*(dim+1),))
    config_id_history = init_config_id.tolist()
    print(f"  Initial config id: {config_id_history}")
    x = all_x[init_config_id]
    y = all_y[init_config_id]
    c = all_c[init_config_id]
    best_y_history = [y.max().item()]
    best_id_history = [config_id_history[y.argmax().item()]]
    cost_history = [0]

    # Initialization for expected minimum simple regret gap stopping rule
    old_model = fit_gp_model(X=x[:-1], objective_X=y[:-1], output_standardize=output_standardize)
    old_config_x = x[-1]

    # # Initialization for probabilistic regret bound (PRB) stopping rule
    # epsilon = 0.005
    # num_samples = 64

    # Independent seed for Thompson sampling
    ts_seed  = seed + 1

    acq_history = {
        'StablePBGI(1e-3)': [np.nan],
        'StablePBGI(1e-4)': [np.nan],
        'StablePBGI(1e-5)': [np.nan],
        'LogEIC': [np.nan],
        'regret upper bound': [np.nan],
        'exp min regret gap': [np.nan],
        # 'PRB': [np.nan]
    }

    for i in range(n_iter):
        # 1. Fit a GP model on the current data.
        model = fit_gp_model(X=x, objective_X=y, output_standardize=output_standardize)
        
        # 2. Determaxe the best observed objective value.
        best_f = y.max()
            
        # 3. Define the acquisition function.
        StablePBGI_1e_3 = StableGittinsIndex(model=model, maximize=maximize, lmbda=1e-3)
        StablePBGI_1e_4 = StableGittinsIndex(model=model, maximize=maximize, lmbda=1e-4)
        StablePBGI_1e_5 = StableGittinsIndex(model=model, maximize=maximize, lmbda=1e-5)
        LogEIC = LogExpectedImprovementWithCost(model=model, best_f=best_f, maximize=maximize)
        beta = 2 * np.log(dim * ((i + 1) ** 2) * (math.pi ** 2) / (6 * 0.1)) / 5
        UCB = UpperConfidenceBound(model=model, maximize=maximize, beta=beta)
        LCB = LowerConfidenceBound(model=model, maximize=maximize, beta=beta)

        # 4. Evaluate the acquisition function on all candidate x's.
        StablePBGI_1e_3_acq = torch.empty(num_configs)
        StablePBGI_1e_4_acq = torch.empty(num_configs)
        StablePBGI_1e_5_acq = torch.empty(num_configs)
        LogEIC_acq = torch.empty(num_configs)
        UCB_acq = torch.empty(num_configs)
        LCB_acq = torch.empty(num_configs)

        with torch.no_grad():
            for j in range(num_batches):
                start = j * batch_size
                end = min((j + 1) * batch_size, num_configs)
                batch_x = all_x[start:end].unsqueeze(1)
                batch_c = estimated_costs[start:end]
                StablePBGI_1e_3_acq[start:end] = StablePBGI_1e_3(batch_x, cost_X=batch_c).squeeze(-1)
                StablePBGI_1e_4_acq[start:end] = StablePBGI_1e_4(batch_x, cost_X=batch_c).squeeze(-1)
                StablePBGI_1e_5_acq[start:end] = StablePBGI_1e_5(batch_x, cost_X=batch_c).squeeze(-1)
                LogEIC_acq[start:end] = LogEIC(batch_x, cost_X=batch_c).squeeze(-1)
                UCB_acq[start:end] = UCB(batch_x).squeeze(-1)
                LCB_acq[start:end] = LCB(batch_x).squeeze(-1)

        # 5. Select the candidate with the optimal acquisition value.
        all_ids = torch.arange(num_configs)
        mask = torch.ones(num_configs, dtype=torch.bool)
        mask[config_id_history] = False
        candidate_ids = all_ids[mask]
        
        if acq == "StablePBGI(1e-3)":
            candidate_acqs = StablePBGI_1e_3_acq[mask]
        if acq == "StablePBGI(1e-4)":
            candidate_acqs = StablePBGI_1e_4_acq[mask]
        if acq == "StablePBGI(1e-5)":
            candidate_acqs = StablePBGI_1e_5_acq[mask]
        if acq == "LogEIC":
            candidate_acqs = LogEIC_acq[mask]
        if acq == "UCB":
            candidate_acqs = UCB_acq[mask]
        if acq == "TS":
            prev_state = torch.get_rng_state()
            torch.manual_seed(ts_seed)
            sample_path = draw_matheron_paths(model, sample_shape=torch.Size([1]))
            torch.set_rng_state(prev_state)
            TS_acq = torch.empty(num_configs)
            with torch.no_grad():
                for j in range(num_batches):
                    start = j * batch_size
                    end = min((j + 1) * batch_size, num_configs)
                    batch_x = all_x[start:end].unsqueeze(1)
                    TS_acq[start:end] = sample_path(batch_x).squeeze()
            candidate_acqs = TS_acq[mask]
            
        new_config_id = candidate_ids[torch.argmax(candidate_acqs)]
        new_config_acq = torch.max(candidate_acqs)
        new_config_x = all_x[new_config_id]
        
        # 6. Query the objective for the new configuration.
        new_config_y = all_y[new_config_id]
        new_config_c = all_c[new_config_id]

        # 7. Record information for stopping.

        # 7.1. Get the posterior mean for old and new GPs at the new and old best points.
        # new_config_x and old_config_x should be the configurations corresponding to the current
        # and previous best indices, respectively.
        x_pair = torch.stack([new_config_x, old_config_x])

        # 7.2. Get posterior mean and covariance from the new model.
        new_posterior = model.posterior(x_pair)
        new_mean = new_posterior.mean         # Shape: [2]
        new_covar = new_posterior.mvn.covariance_matrix     # Shape: [2, 2]

        # 7.3. Get posterior mean and covariance from the old model.
        old_posterior = old_model.posterior(x_pair)
        old_mean = old_posterior.mean           # Shape: [2]
        old_covar = old_posterior.mvn.covariance_matrix       # Shape: [2, 2]

        # 7.4. Compute delta_mu (the absolute change in best posterior mean)
        # Here, we assume that new_config_x corresponds to the current best (new point)
        # and old_config_x corresponds to the previous best.
        delta_mu = abs(old_mean[1].item() - new_mean[0].item())

        # 7.5. Compute κ_{t−1} = UCB - LCB gap.
        kappa = torch.max(UCB_acq) - torch.max(LCB_acq[~mask])

        # 7.6. Compute KL divergence between old and new posteriors at the new point.
        old_var = old_covar[0, 0].clamp(max=1e-12)
        new_var = new_covar[0, 0].clamp(max=1e-12)
        old_mu_val = old_mean[0]
        new_mu_val = new_mean[0]
        kl = 0.5 * (torch.log(new_var / old_var) +
                    (old_var + (old_mu_val - new_mu_val).pow(2)) / new_var - 1).item()

        # 7.7. Compute ei_diff, the expected-improvement gap difference.
        # If new_config_x and old_config_x are (approximately) equal, we set ei_diff to zero.
        if not torch.allclose(new_config_x, old_config_x, atol=1e-6):
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
        acq_history['exp min regret gap'].append(exp_min_regret_gap)
        acq_history['regret upper bound'].append(kappa.item())

        # 7.9. Reassign old_model and old_config_x for the next iteration.
        old_model = model
        old_config_x = new_config_x

        # Other stopping rules
        acq_history['StablePBGI(1e-3)'].append(torch.max(StablePBGI_1e_3_acq).item())
        acq_history['StablePBGI(1e-4)'].append(torch.max(StablePBGI_1e_4_acq).item())
        acq_history['StablePBGI(1e-5)'].append(torch.max(StablePBGI_1e_5_acq).item())
        acq_history['LogEIC'].append(torch.max(LogEIC_acq[mask]).item())
        
        # 8. Append the new data to our training set.
        x = torch.cat([x, new_config_x.unsqueeze(0)], dim=0)
        y = torch.cat([y, new_config_y.unsqueeze(0)], dim=0)
        c = torch.cat([c, new_config_c.unsqueeze(0)], dim=0)
        config_id_history.append(new_config_id.item())
        best_y_history.append(best_f.item())
        best_id_history.append(config_id_history[y.argmax().item()])
        cost_history.append(new_config_c.item())

        print(f"Iteration {i + 1}:")
        print(f"  Selected config_id: {new_config_id}")
        print(f"  Acquisition value: {new_config_acq.item():.4f}")
        print(f"  Objective (final_accuracy): {new_config_y.item():.4f}")
        print(f"  Cost (time): {new_config_c.item():.4f}")
        print(f"  Current best observed: {best_f.item():.4f}")
        print()

        del StablePBGI_1e_3, StablePBGI_1e_4, StablePBGI_1e_5
        del LogEIC, UCB, LCB
        gc.collect()

    best_y_history.append(y.max().item())

    # Return the history including the acq_history dictionary.
    return (cost_history,
            [best_id_history[0]] + config_id_history[-n_iter:], 
            best_id_history,
            best_y_history,
            acq_history)


wandb.init(reinit=True, sync_tensorboard=False, settings=wandb.Settings(_disable_stats=True))

result = run_bayesopt_experiment(wandb.config)

(cost_history, config_id_history, best_id_history, best_y_history, acq_history) = result

cumulative_costs = np.cumsum(cost_history)

# Log full info
for idx in range(len(cost_history)):
    log_dict = {
        "config id": config_id_history[idx],
        "cumulative cost": cumulative_costs[idx],
        "current best id": best_id_history[idx],
        "current best observed": best_y_history[idx],
        "StablePBGI(1e-3) acq": acq_history['StablePBGI(1e-3)'][idx],
        "StablePBGI(1e-4) acq": acq_history['StablePBGI(1e-4)'][idx],
        "StablePBGI(1e-5) acq": acq_history['StablePBGI(1e-5)'][idx],
        "LogEIC acq": acq_history['LogEIC'][idx],
        "exp min regret gap": acq_history['exp min regret gap'][idx],
        "regret upper bound": acq_history['regret upper bound'][idx],
        # "PRB": acq_history['PRB'][idx]
    }
    wandb.log(log_dict)
    time.sleep(0.5)  # Delay of 0.5s per entry

wandb.finish()