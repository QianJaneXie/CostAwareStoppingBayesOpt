print("=== STARTING sss_known_cost.py ===", flush=True)

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
import argparse
import json

# Set default tensor type to float64
torch.set_default_dtype(torch.float64)

import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

def run_bayesopt_experiment(bayesopt_config):
    print(bayesopt_config, flush=True)
    dataset_name = bayesopt_config['dataset_name']
    seed = bayesopt_config['seed']
    output_standardize = bayesopt_config['output_standardize']
    maximize = bayesopt_config['maximize']
    dim = bayesopt_config['dim']
    n_iter = bayesopt_config['num_iteration']
    num_configs = bayesopt_config['num_configs']
    acq = bayesopt_config['acquisition_function']

    # Load configuration data files using absolute paths
    all_x = torch.load(os.path.join(script_dir, 'NATS_data/sss_data', 'nats_all_x.pt'))
    all_flops = torch.load(os.path.join(script_dir, 'NATS_data/sss_data', 'nats_all_flops.pt'))
    all_y = torch.load(os.path.join(script_dir, 'NATS_data/sss_data', f'nats_{dataset_name}_all_y.pt'))
    all_c = torch.load(os.path.join(script_dir, 'NATS_data/sss_data', f'nats_{dataset_name}_all_c.pt'))

    # Compute estimated runtime based on flops
    if dataset_name == "cifar10-valid":
        alpha = 1
        beta = 400
    if dataset_name == "cifar100":
        alpha = 2
        beta = 550
    if dataset_name == "ImageNet16-120":
        alpha = 1
        beta = 1000
    estimated_costs = alpha * all_flops + beta

    # Sample initial configurations
    torch.manual_seed(seed)
    init_config_id = torch.randint(low=0, high=num_configs, size=(2*(dim+1),))
    config_id_history = init_config_id.tolist()
    print(f"  Initial config id: {config_id_history}")
    x = all_x[init_config_id]
    y = all_y[init_config_id]
    c = all_c[init_config_id]
    best_y_history = [y.min().item()]
    best_id_history = [config_id_history[y.argmin().item()]]
    cost_history = [0]
    estimated_cost_history = [0]

    # Initialization for expected minimum simple regret gap stopping rule
    old_model = fit_gp_model(X=x[:-1], objective_X=y[:-1], output_standardize=output_standardize)
    old_config_x = x[-1]

    # Initialization for probabilistic regret bound (PRB) stopping rule
    if dataset_name == "cifar10-valid":
        epsilon = 0.48
    if dataset_name == "cifar100":
        epsilon = 1.43
    if dataset_name == "ImageNet16-120":
        epsilon = 2.63
    num_samples = 64

    # Independent seed for Thompson sampling
    ts_seed  = seed + 1

    acq_history = {
        'StablePBGI(1e-3)': [np.nan],
        'StablePBGI(1e-4)': [np.nan],
        'StablePBGI(1e-5)': [np.nan],
        'LogEIC': [np.nan],
        'regret upper bound': [np.nan],
        'exp min regret gap': [np.nan],
        'PRB': [np.nan]
    }

    for i in range(n_iter):
        # 1. Fit a GP model on the current data.
        model = fit_gp_model(X=x, objective_X=y, output_standardize=output_standardize)
        
        # 2. Determine the best observed objective value.
        best_f = y.min()
            
        # 3. Define the acquisition function.
        StablePBGI_1e_3 = StableGittinsIndex(model=model, maximize=maximize, lmbda=1e-3)
        StablePBGI_1e_4 = StableGittinsIndex(model=model, maximize=maximize, lmbda=1e-4)
        StablePBGI_1e_5 = StableGittinsIndex(model=model, maximize=maximize, lmbda=1e-5)
        LogEIC = LogExpectedImprovementWithCost(model=model, best_f=best_f, maximize=maximize)
        beta = 2 * np.log(dim * ((i + 1) ** 2) * (math.pi ** 2) / (6 * 0.1)) / 5
        UCB = UpperConfidenceBound(model=model, maximize=maximize, beta=beta)
        LCB = LowerConfidenceBound(model=model, maximize=maximize, beta=beta)

        # 4. Evaluate the acquisition function on all candidate x's.
        StablePBGI_1e_3_acq = StablePBGI_1e_3.forward(all_x.unsqueeze(1), cost_X = estimated_costs)
        StablePBGI_1e_3_acq[config_id_history] = y.squeeze(-1)
        StablePBGI_1e_4_acq = StablePBGI_1e_4.forward(all_x.unsqueeze(1), cost_X = estimated_costs)
        StablePBGI_1e_4_acq[config_id_history] = y.squeeze(-1)
        StablePBGI_1e_5_acq = StablePBGI_1e_5.forward(all_x.unsqueeze(1), cost_X = estimated_costs)
        StablePBGI_1e_5_acq[config_id_history] = y.squeeze(-1)
        LogEIC_acq = LogEIC.forward(all_x.unsqueeze(1), cost_X = estimated_costs)
        UCB_acq = UCB.forward(all_x.unsqueeze(1))
        LCB_acq = LCB.forward(all_x.unsqueeze(1))

        # 5. Select the candidate with the optimal acquisition value.
        all_ids = torch.arange(num_configs)
        mask = torch.ones(num_configs, dtype=torch.bool)
        mask[config_id_history] = False
        candidate_ids = all_ids[mask]
        
        if acq == "StablePBGI(1e-3)":
            candidate_acqs = StablePBGI_1e_3_acq[mask]
            new_config_id = candidate_ids[torch.argmin(candidate_acqs)]
            new_config_acq = torch.min(candidate_acqs)
        if acq == "StablePBGI(1e-4)":
            candidate_acqs = StablePBGI_1e_4_acq[mask]
            new_config_id = candidate_ids[torch.argmin(candidate_acqs)]
            new_config_acq = torch.min(candidate_acqs)
        if acq == "StablePBGI(1e-5)":
            candidate_acqs = StablePBGI_1e_5_acq[mask]
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
        if acq == "TS":
            prev_state = torch.get_rng_state()
            torch.manual_seed(ts_seed)
            sample_path = draw_matheron_paths(model, sample_shape=torch.Size([1]))
            torch.set_rng_state(prev_state)
            TS_acq = sample_path(all_x).squeeze()
            candidate_acqs = TS_acq[mask]
            new_config_id = candidate_ids[torch.argmin(candidate_acqs)]
            new_config_acq = torch.min(candidate_acqs)

        new_config_x = all_x[new_config_id]
        
        # 6. Query the objective for the new configuration.
        new_config_y = all_y[new_config_id]
        new_config_c = all_c[new_config_id]
        new_config_estimated_c = estimated_costs[new_config_id]

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
        kappa = torch.min(UCB_acq[~mask]) - torch.min(LCB_acq)

        # 7.6. Compute KL divergence between old and new posteriors at the new point.
        old_var = old_covar[0, 0].clamp(min=1e-12)
        new_var = new_covar[0, 0].clamp(min=1e-12)
        old_mu_val = old_mean[0]
        new_mu_val = new_mean[0]
        kl = 0.5 * (torch.log(new_var / old_var) +
                    (old_var + (old_mu_val - new_mu_val).pow(2)) / new_var - 1).item()

        # 7.7. Compute ei_diff, the expected-improvement gap difference.
        # If new_config_x and old_config_x are (approximately) equal, we set ei_diff to zero.
        if not torch.allclose(new_config_x, old_config_x, atol=1e-4):
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

        # Probabilistic regret bound
        paths = draw_matheron_paths(model, sample_shape=torch.Size([num_samples]))
        best_x = all_x[config_id_history[y.argmin().item()]]
        regrets = paths(best_x.unsqueeze(0)).squeeze(-1) - paths(all_x).min(dim=1).values
        prb_estimate = (regrets <= epsilon).float().mean().item()
        acq_history['PRB'].append(prb_estimate)
        num_samples = min(math.ceil(num_samples * 1.5), 1000)

        # Other stopping rules
        acq_history['StablePBGI(1e-3)'].append(torch.min(StablePBGI_1e_3_acq).item())
        acq_history['StablePBGI(1e-4)'].append(torch.min(StablePBGI_1e_4_acq).item())
        acq_history['StablePBGI(1e-5)'].append(torch.min(StablePBGI_1e_5_acq).item())
        acq_history['LogEIC'].append(torch.max(LogEIC_acq[mask]).item())
        
        # 8. Append the new data to our training set.
        x = torch.cat([x, new_config_x.unsqueeze(0)], dim=0)
        y = torch.cat([y, new_config_y.unsqueeze(0)], dim=0)
        c = torch.cat([c, new_config_c.unsqueeze(0)], dim=0)
        config_id_history.append(new_config_id.item())
        best_y_history.append(best_f.item())
        best_id_history.append(config_id_history[y.argmin().item()])
        cost_history.append(new_config_c.item())
        estimated_cost_history.append(new_config_estimated_c.item())

        print(f"Iteration {i + 1}:", flush=True)
        print(f"  Selected config_id: {new_config_id}", flush=True)
        print(f"  Acquisition value: {new_config_acq.item():.4f}", flush=True)
        print(f"  Objective (validation error): {new_config_y.item():.4f}", flush=True)
        print(f"  Cost (runtime): {new_config_c.item():.4f}", flush=True)
        print(f"  Estimated cost (alpha * flops + beta): {new_config_estimated_c.item():.4f}", flush=True)
        print(f"  Current best observed: {best_f.item():.4f}", flush=True)
        print()

        del StablePBGI_1e_3, StablePBGI_1e_4, StablePBGI_1e_5
        del LogEIC, UCB, LCB
        gc.collect()

    best_y_history.append(y.min().item())

    # Return the history including the acq_history dictionary.
    return (cost_history,
            estimated_cost_history,
            [best_id_history[0]] + config_id_history[-n_iter:], 
            best_id_history,
            best_y_history,
            acq_history)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to JSON config file')
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.getcwd(), config_path)

    print(f"Resolved config path: {config_path}", flush=True)

    with open(config_path, 'r') as f:
        bayesopt_config = json.load(f)

    result = run_bayesopt_experiment(bayesopt_config)

    cost_history, estimated_cost_history, config_id_history, best_id_history, best_y_history, acq_history = result

    output_dir = os.path.join(script_dir, 'NATS_results/sss_results')
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{bayesopt_config['dataset_name']}_seed{bayesopt_config['seed']}_{bayesopt_config['acquisition_function'].replace(' ', '').replace('(', '').replace(')', '')}.jsonl"
    output_path = os.path.join(output_dir, filename)

    # Prepare the full dict once
    result_dict = {
        "config id": list(map(int, config_id_history)),
        "cumulative cost": list(map(float, np.cumsum(cost_history))),
        "estimated cumulative cost": list(map(float, np.cumsum(estimated_cost_history))),
        "current best id": list(map(int, best_id_history)),
        "current best observed": list(map(float, best_y_history)),
        "StablePBGI(1e-3) acq": list(map(float, acq_history.get('StablePBGI(1e-3)', [np.nan]*len(cost_history)))),
        "StablePBGI(1e-4) acq": list(map(float, acq_history.get('StablePBGI(1e-4)', [np.nan]*len(cost_history)))),
        "StablePBGI(1e-5) acq": list(map(float, acq_history.get('StablePBGI(1e-5)', [np.nan]*len(cost_history)))),
        "LogEIC acq": list(map(float, acq_history.get('LogEIC', [np.nan]*len(cost_history)))),
        "exp min regret gap": list(map(float, acq_history.get('exp min regret gap', [np.nan]*len(cost_history)))),
        "regret upper bound": list(map(float, acq_history.get('regret upper bound', [np.nan]*len(cost_history)))),
        "PRB": list(map(float, acq_history.get('PRB', [np.nan]*len(cost_history))))
    }

    # Write to JSON file
    with open(output_path, 'w') as f:
        json.dump(result_dict, f, indent=2)

    print(f"Saved results to {output_path}")

    done_dir = os.path.join(script_dir, 'NATS_results', 'sss_known_done')
    os.makedirs(done_dir, exist_ok=True)

    done_filename = os.path.basename(output_path).replace('.jsonl', '.done')
    done_path = os.path.join(done_dir, done_filename)

    with open(done_path, 'w') as f:
        f.write("success\n")



if __name__ == '__main__':
    main()
