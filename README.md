
<div align="center">

[![Generic badge](https://img.shields.io/badge/The%20Lancet%20Digital%20Health-Fulltext-3700B3)](https://www.thelancet.com/journals/landig/article/PIIS2589-7500(22)00070-X/fulltext)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Generic badge](https://img.shields.io/badge/Website-Vara-7273fb)](https://www.vara.ai)

</div>

### Are all clinical decisions equal? A retrospective analysis combining the strengths of radiologists and AI for breast cancer screening
*Christian Leibig, Moritz Brehmer, Stefan Bunk, Danalyn Byng, Katja Pinker, Lale Umutlu; The Lancet Digital Health, 2022*

Code and necessary data to reproduce figures and tables from the paper.

### Brief summary
`normal_triaging` models perform predictions on studies that are considered very likely negative and `safety_net` 
models perform predictions on studies that are considered very likely positive. All other decisions are referred to 
radiologists. This code analyses the impact decision referral has on the radiologists' performance when radiologists 
and AI collaborate as proposed. Both the average as well as subgroup performance is assessed.

### Data
* All necessary input data should be placed under `data/inputs`. We provide one pandas dataframe stored in HDF5 for each of 
  the [internal validation](https://storage.googleapis.com/mx-healthcare-pub/internal_validation_set.h5), 
  [internal test](https://storage.googleapis.com/mx-healthcare-pub/internal_test_set.h5) and 
  [external test](https://storage.googleapis.com/mx-healthcare-pub/external_test_set.h5) datasets (see Figure 2 in the paper).
  [Running the pipeline](#generating-figures-and-tables) automatically downloads the data into the expected location.
* All outputs are stored under `data/results`.

### Setup and dependencies
Setup happens automatically upon first execution of the [pipeline](#generating-figures-and-tables):
```bash
git clone https://github.com/vara-ai/decision-referral.git
# From repository root dir:
./run_pipeline.sh minimal
```
This sets up the environment by downloading a docker image and the necessary input data. Alternatively, you can build 
the docker image locally via 
```bash
docker build -t vara_dr .
```
or just setup a python (3.8) environment: `pip install -r requirements.txt`.

### Generating figures and tables
To check that everything is setup correctly:
```bash
./run_pipeline.sh minimal
```
to run all [steps](#entrypoints), but just for a single configuration and without any statistical tests to speed things 
up. This will set thresholds on validation data, compute decision referral on internal as well as external test data and 
generate the corresponding figures and tables. This should take ~ 1 minute on a standard laptop and generate 
output like:
```bash
data/results
└── internal_validation_set
    ├── decision_referral
    │   ├── config-20220315-163705-38c1.yaml
    │   ├── decision_referral_result.pkl
    │   └── ...
    ├── plots
    │   ├── config-20220315-163716-de67.yaml
    │   ├── results_table.csv
    │   ├── roc_curve_subset_assessed_by_ai_(97.0%,_98.0%).png
    │   ├── subgroup_sensitivities_nt@0.97+sn@0.98.csv
    │   ├── subgroup_sensitivities_nt@0.97+sn@0.98.png
    │   ├── subgroup_specificities_nt@0.97+sn@0.98.csv
    │   ├── system_performance.png
    │   └── ...
    └── test_sets
        ├── external_test_set
        │   ├── decision_referral
        │   │   ├── config-20220315-163754-9ad1.yaml
        │   │   ├── decision_referral_result.pkl
        │   │   ├── ...
        │   └── plots
        │       ├── config-20220315-163811-64d6.yaml
        │       ├── results_table.csv
        │       ├── roc_curve_subset_assessed_by_ai_(97.0%,_98.0%).png
        │       ├── subgroup_sensitivities_nt@0.97+sn@0.98.csv
        │       ├── subgroup_sensitivities_nt@0.97+sn@0.98.png
        │       ├── subgroup_specificities_nt@0.97+sn@0.98.csv
        │       ├── system_performance.png
        │       └── ...
        └── internal_test_set
            ├── decision_referral
            │   ├── config-20220315-163705-a8e5.yaml
            │   ├── decision_referral_result.pkl
            │   ├── ...
            └── plots
                ├── config-20220315-163723-d4eb.yaml
                ├── results_table.csv
                ├── roc_curve_subset_assessed_by_ai_(97.0%,_98.0%).png
                ├── subgroup_sensitivities_nt@0.97+sn@0.98.csv
                ├── subgroup_sensitivities_nt@0.97+sn@0.98.png
                ├── subgroup_specificities_nt@0.97+sn@0.98.csv
                ├── system_performance.png
                └── ...
 ```

The full results from the publication can be reproduced via:
```bash
./run_pipeline maximal
```

### Package layout

`decision_referral/core.py`: Provides the core logic of assembling predictions from different models and radiologists, 
    including threshold setting, assessing statistical significance, etc. This file cannot be called on its own.

#### Entrypoints
`evaluate.py`: Performs the decision referral computation for the given models on the given dataset(s), 
storing all artefacts needed by subsequent steps. This function _must_ be run on a given dataset before any other 
scripts will work.

`generate_plots.py`: Using the artefacts from `evaluate.py` this script generates tables and figures.

#### Parametrization
We use [hydra](https://hydra.cc/docs/intro/) for configuration. Adapt `conf/evaluate.yaml` and/or 
`conf/generate-plots.yaml` for the two entry points described [above](#entrypoints). Alternatively, you can overwrite
parameters via command-line flags as demonstrated in `run_pipeline.sh`.

### Hardware requirements and performance
Regenerating all results requires <=16GB RAM for the internal test set and <=64GB for the external test set. Most 
expensive is `evaluate.py` and therein the resampling based hypothesis tests which come with a vectorized 
implementation. The decision referral operating pairs from the `maximal` configuration for full 
reproducibility can be computed in parallel if sufficient RAM is available. Without any parallelization the whole 
pipeline should run in ~1h - 2h. `evaluate.py` does the bulk of the work, `generate_plots.py` just takes a fraction of 
the time.

### Reproducibility
Since the results had been frozen for peer review, our implementations slightly evolved (e.g. the threshold setting had 
been vectorised in the meantime), leading to very slight deviations in the reported results, not affecting any 
interpretation of results. Apart from that, the resampling based CIs and exact p-values will again vary very slightly 
between runs, reflecting the stochastic nature of the underlying method. Of course our models have also evolved since 
peer-review, but we'll keep that reporting to future work, stay tuned.

### Contributors

Main contributors for the paper and verifying correctness:
* Christian Leibig
* Stefan Bunk

Contributors (in alphabetical order):
* Benjamin Strauch
* Christian Leibig
* Dominik Schüler
* Michael Ball
* Stefan Bunk
* Vilim Štih
* Zacharias Fisches

### Contact
At [Vara](vara.ai), we are committed to reproducible research. Please feel free to reach out to the corresponding author
([firstname] [dot] [lastname] [at] vara.ai) if you have trouble reproducing results or any questions about the decision 
referral approach.
