# CostAwareStoppingBayesOpt
 Cost-aware Stopping for Bayesian Optimization on Bayesian regret and AutoML Benchmarks

## Combinations of acquisiton function and stopping rules
- **Acquisition fuctions**:
  - the Pandora's Box Gittins index (PBGI)
  - log expected improvement per cost (LogEIPC)
  - lower confidence bound (LCB)
  - Thompson sampling (TS)
- **Our stopping rules**:
  - PBGI/LogEIPC
- **Baseline stopping rules**:
  - LogEIPC-med
  - gap of expected minimum simple regrets (SRGap-med)
  - UCB-LCB
  - probabilistic regret bound (PRB)
  - global stopping strategy (GSS)
  - Convergence
  - Hindsight 

The implementation of PBGI, LogEIPC, and LCB can be found in `pandora_automl/acquisition`.
 
## Contexts
- **Experiments**
  - 1D Bayesian regret
  - 8D Bayesian regret (continuous optimizer)
  - Empirical (LCBench)
    - Known cost
    - Unknown cost 
  - Timing
- **Illustrations**
  - Stopping behaviors of PBGI and LogEIPC under large and small evaluation costs
