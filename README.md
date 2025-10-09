# CostAwareStoppingBayesOpt
 Cost-aware Stopping for Bayesian Optimization on Bayesian regret and AutoML Benchmarks

## Combinations of acquisiton function and stopping rules
- **Acquisition fuctions**:
  - the Pandora's Box Gittins index (PBGI)
  - log expected improvement per cost (LogEIPC)
  - lower confidence bound (LCB)
  - Thompson sampling (TS)
- **Our stopping rule**:
  - PBGI/LogEIPC
- **Baseline stopping rules**:
  - LogEIPC-med
  - UCB-LCB
  - gap of expected minimum simple regrets (SRGap-med)
  - probabilistic regret bound (PRB)
  - global stopping strategy (GSS)
  - Convergence
  - Hindsight 

The implementation of PBGI, LogEIPC, and LCB can be found in `pandora_automl/acquisition`.
 
## Contexts
- **Experiments**
  - 1D Bayesian regret
  - 8D Bayesian regret (continuous optimizer)
  - Empirical (LCBench, NATS-Bench)
    - Known cost (proxy time, actual time)
    - Unknown cost 
  - Timing
- **Illustrations**
  - Stopping behaviors of PBGI and LogEIPC under large and small uniform evaluation costs
  - Smoothing (moving average) of LogEIPC acquisition values in 8D Bayesian regret experiments to avoid wiggles due to imperfect continuous acquisition function optimization
  - Comparison of two variants of PBGI stopping rules: the before-posterior-update (this-round) stopping
rule and the after-posterior-update (next-round) stopping rule.
