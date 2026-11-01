# LegendMachineLearning.jl

[![Documentation for stable version](https://img.shields.io/badge/docs-stable-blue.svg)](https://legend-exp.github.io/LegendMachineLearning.jl/stable)
[![Documentation for development version](https://img.shields.io/badge/docs-dev-blue.svg)](https://legend-exp.github.io/LegendMachineLearning.jl/dev)
[![Build Status](https://github.com/legend-exp/LegendMachineLearning.jl/workflows/CI/badge.svg)](https://github.com/legend-exp/LegendMachineLearning.jl/actions/workflows/CI.yml)
[![Codecov](https://codecov.io/gh/legend-exp/LegendMachineLearning.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/legend-exp/LegendMachineLearning.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)


## Documentation

* [Documentation for stable version](https://legend-exp.github.io/LegendMachineLearning.jl/stable)
* [Documentation for development version](https://legend-exp.github.io/LegendMachineLearning.jl/dev)

## QC classification pipeline

```julia
using LegendMachineLearning, LegendDataManagement

l200 = LegendData(:l200)
fk = training_filekey(l200, DataPeriod(3), DataCategory(:cal))
config = dataprod_config(l200).qc(fk).ml

# 1. sample + affinity propagation (writes the jlqml sample file)
sample, cluster, pars, scan = run_qc_sampling(l200, fk, config)

# 2. label the exemplars (pars.exemplar_labels[k] for each cluster k), e.g. in the Pluto notebook

# 3. train, tune and store the model (writes the jld2 model file)
model, pars_entry, rep = run_qc_training(l200, fk, config, pars)

# 4. classify waveforms
labels = model(waveforms)
```

The config lives under `dataprod_config(l200).qc(filekey).ml`, with keys
`detectors`, `n_max`, `waveform_column`, `strata`, `features`,
`affinity_propagation`, `svm`, and `categories`. The sample is drawn from the
raw files of every analysis run of the period in two passes: a scan of the DAQ
row scalars, then `waveform_stats` on the waveforms of `n_scan` pool rows per
detector, from which the rules draw their strata:

```yaml
strata:
  aux_triggers: {pulser: PULS01, forced_trigger: BSLN01}
  exclude: [pulser, forced_trigger]
  ts_window: 125.0
  n_scan: 200000
  bl_window: [0.0, 39.0]      # µs
  tail_window: [70.0, 95.0]   # µs
  rules:
    - {name: zero_energy, select: "daqenergy == 0", fraction: 0.10}
    - {name: saturated, select: "t_sat_hi > 0 || t_sat_lo > 0", fraction: 0.05}
    - {name: bad_tau, select: "daqenergy > 0 && !(tail_tau > 150us && tail_tau < 2000us)", fraction: 0.10}
    - {name: multi_trigger, select: "n_triggers > 1", fraction: 0.05}
    - {name: baseline_outlier, select: "baseline_dev > 5", fraction: 0.05}
  uniform: "daqenergy > 0"
```

`pars_entry` is what the processor writes to `rpars.ml`, together with a
validity entry for the training run.

## The `legend-qcml` app

`apps/LegendMLApps` is a separate package that serves a Pluto notebook for the
labeling, training and check steps: it reads the sample and the `rpars.ml` entry
of a period and category, lets you label the cluster exemplars and save them to
the entry, trains the SVM (model file next to the sample, entry and validity
updated), and classifies the raw events of any file of the period with the
stored model. It needs Julia 1.12 or later.

Install it from a checkout of this repository, so that changes to the checkout
take effect without reinstalling:

```
pkg> app develop path/to/LegendMachineLearning/apps/LegendMLApps
```

or from GitHub:

```julia
using Pkg
Pkg.Apps.add(url="https://github.com/legend-exp/LegendMachineLearning.jl", subdir="apps/LegendMLApps")
```

The app is installed as `~/.julia/bin/legend-qcml` (`$JULIA_DEPOT_PATH/bin` with a
non-default depot); put that directory on your `PATH`, e.g. in `~/.zshrc`:

```sh
export PATH="$HOME/.julia/bin:$PATH"
```

Usage:

```
legend-qcml [--config PATH] [--period PERIOD] [--category CATEGORY] [--port PORT] [--host HOST]

  --config PATH        LEGEND data config (default: $LEGEND_DATA_CONFIG)
  --period PERIOD      period selected at start, e.g. p18 (default: the last period with an entry)
  --category CATEGORY  cal or phy (default: cal)
  --port PORT          port of the Pluto server (default: 10004)
  --host HOST          address the server listens on (default: 127.0.0.1)
```

The app stops with an error when the config is missing or the period has no
entry. It runs Julia with 8 threads; `legend-qcml --threads=4 -- --period p18`
overrides that. To work on a remote machine, start the app there (e.g. in a
`tmux` session) and forward the port from your own machine:

```sh
ssh cslg4                       # then, on cslg4:
legend-qcml --period p18 --category cal
ssh -L 10004:localhost:10004 cslg4   # on your own machine
```

and open <http://localhost:10004/> in the browser.
