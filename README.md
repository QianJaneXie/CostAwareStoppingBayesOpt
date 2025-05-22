# CostAwareStoppingBayesOpt
 Cost-aware Stopping for Bayesian Optimization on Bayesian regret and AutoML Benchmarks

## Combinations of acquisiton function and stopping rules
- **Acquisition fuctions**:
  - PBGI
  - LogEIC
  - LCB
  - TS
- **Our stopping rules**:
  - PBGI/LogEIC
- **Baseline stopping rules**:
  - LogEIC-med
  - SRGap-med
  - UCB-LCB
  - PRB
  - GSS
  - Convergence
  - Hindsight 

The implementation of PBGI, LogEIC, and LCB can be found in `pandora_automl/acquisition`.
 
## Contexts
- **Experiments**
  - 1D Bayesian regret
  - 8D Bayesian regret (continuous optimizer)
  - Empirical (LCBench)
  - Timing
- **Illustrations**
  - Stopping behaviors of PBGI and EIPC under large and small evaluation costs