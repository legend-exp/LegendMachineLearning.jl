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
raw files of every analysis run of the period, in strata:

```yaml
strata:
  aux_triggers: {pulser: PULS01, forced_trigger: BSLN01}
  exclude: [pulser, forced_trigger]
  zero_energy: 0.2
  high_energy: 0.15
  high_energy_quantile: 0.9
  baseline_outlier: 0.15
  baseline_mad: 5.0
```

`pars_entry` is what the processor writes to `rpars.ml`, together with a
validity entry for the training run.
