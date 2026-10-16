# Wearable Stress Detection: Physiological Signals and CNN-Based Feature Learning

Research code for binary **stress vs. rest classification** using
physiological signals from the [Wearable Device Dataset from Induced
Stress and Structured Exercise Sessions (PhysioNet
v1.0.1)](https://physionet.org/content/wearable-device-dataset/1.0.1/).

The repository contains a paper-aligned handcrafted-feature baseline and
CNN-based feature-learning experiments. It is intended for research and
reproducibility; results must be generated locally and should not be
assumed to match published metrics exactly.

## Dataset

**Source:** Hongn, A., Bosch, F., Prado, L., & Bonomini, P. (2025).
*Wearable Device Dataset from Induced Stress and Structured Exercise
Sessions* (version 1.0.1). PhysioNet. https://doi.org/10.13026/he0v-tf17

**Original publication:** Hongn, A., Bosch, F., Prado, L. E., Ferrández,
J. M., & Bonomini, M. P. (2025). *Wearable Physiological Signals under
Acute Stress and Exercise Conditions*. Scientific Data, 12(1).
https://doi.org/10.1038/s41597-025-04845-9

The dataset includes Empatica E4 recordings from stress-induction and
structured exercise sessions. This repository's current stress/rest
pipelines use the `STRESS` recordings, rather than claiming to classify
all exercise conditions.

PhysioNet reports recordings from 36 healthy volunteers for stress
sessions, 30 for aerobic exercise, and 31 for anaerobic exercise.
Signals include blood volume pulse (BVP), electrodermal activity (EDA),
heart rate (HR), inter-beat intervals (IBI), skin temperature (TEMP),
three-axis accelerometry (ACC), and protocol event tags.

### Download and place the data

Download the dataset from the [PhysioNet dataset
page](https://physionet.org/content/wearable-device-dataset/1.0.1/).
Extract it locally. Do **not** commit the raw dataset to this GitHub
repository; keep it in a local data directory and follow the dataset's
license and attribution terms.

Expected layout (the dataset root is the folder containing
`Wearable_Dataset/` or, depending on how you organize it, the folder
containing `STRESS/`):

``` text
project/
├── README.md
├── requirements.txt
├── stress_rest_pipleine_v3.py
├── stress_rest_pipleine_v2.py
├── cnn_baseline_feature_comparison.py
├── cnn_stress_contribution_v2.py
└── data/
    └── Wearable_Dataset/
        ├── STRESS/
        │   ├── S01/
        │   │   ├── ACC.csv
        │   │   ├── BVP.csv
        │   │   ├── EDA.csv
        │   │   ├── HR.csv
        │   │   ├── IBI.csv
        │   │   ├── TEMP.csv
        │   │   └── tags.csv
        │   └── ...
        ├── subject-info.csv
        └── ...
```

Pass the path that **directly contains `STRESS/`** to `--root`. For the
example layout above, that is `data/Wearable_Dataset`.

## Repository scripts

  --------------------------------------------------------------------------
  Script                                 Purpose
  -------------------------------------- -----------------------------------
  `stress_rest_pipleine_v3.py`           Paper-aligned handcrafted
                                         stress/rest feature extraction and
                                         classical ML baseline.

  `stress_rest_pipleine_v2.py`           Earlier version of the
                                         handcrafted-feature baseline.

  `cnn_baseline_feature_comparison.py`   Focused CNN feature-learning
                                         comparison, with optional
                                         comparison against the handcrafted
                                         feature CSV.

  `cnn_stress_contribution_v2.py`        CNN feature-learning experiments
                                         for subject-mixed CV,
                                         leave-one-subject-out evaluation,
                                         and protocol-version transfer.
  --------------------------------------------------------------------------

The filenames preserve the spelling used in the supplied code
(`pipleine`).

## Method overview

### 1. Handcrafted-feature baseline

The paper-aligned pipeline:

1.  Reads Empatica CSV signals and event tags.
2.  Segments the stress protocol into baseline, task, and rest blocks.
3.  Uses the second half of `First_Rest` and `Second_Rest` for rest
    examples.
4.  Filters BVP and EDA and extracts a 13-feature set based on Table 4
    of the associated paper.
5.  Applies intra-participant normalization.
6.  Evaluates classical classifiers with stratified cross-validation and
    random oversampling applied within the training pipeline.
7.  Saves metrics, confusion matrices, feature/missingness information,
    and run metadata.

The implementation documents approximations where the paper does not
report every low-level parameter. Therefore, it is a **paper-aligned
reproduction**, not a guaranteed bit-for-bit replication of the
published scores.

### 2. CNN feature-learning pipeline

The CNN scripts build multimodal windows from ACC, BVP, EDA, and HR. By
default, windows are 30 seconds long with a 15-second stride (50%
overlap). A 1D CNN learns window embeddings; embeddings are then
aggregated by protocol block and evaluated with classical ML
classifiers.

Available experiments in `cnn_stress_contribution_v2.py`:

-   `mixed10`: subject-mixed stratified 10-fold block-level
    cross-validation.
-   `loso`: leave-one-subject-out evaluation.
-   `v1_to_v2`: train on protocol v1 (`Sxx`) and test on protocol v2
    (`fxx`).
-   `v2_to_v1`: train on protocol v2 (`fxx`) and test on protocol v1
    (`Sxx`).
-   `all`: run all supported experiments.

For LOSO and cross-version transfer, CNN training is performed within
each evaluation split so that the held-out test subjects/blocks are not
used to train the CNN representation.

**Evaluation caution:** subject-mixed block-level cross-validation is
not equivalent to subject-independent validation. Prefer LOSO or clearly
defined cross-version transfer when making claims about generalization
to unseen participants or protocols. Overlapping windows must not be
treated as independent participants.

## Environment setup

Python 3.10 or 3.11 is a reasonable starting point. Install a PyTorch
build suitable for your CPU/GPU from the [official PyTorch installation
selector](https://pytorch.org/get-started/locally/), then install the
remaining dependencies.

Example:

``` bash
python -m venv .venv
```

Activate the environment:

**Windows PowerShell**

``` powershell
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS**

``` bash
source .venv/bin/activate
```

Install packages:

``` bash
python -m pip install --upgrade pip
python -m pip install numpy pandas scipy scikit-learn imbalanced-learn xgboost neurokit2 torch
```

`xgboost` and `neurokit2` are optional in parts of the code; installing
them enables the corresponding model or signal-processing paths. Some
scripts require PyTorch and `imbalanced-learn`.

## Run the experiments

Run commands from the repository root. Replace the example dataset path
with your actual local path.

### Handcrafted-feature baseline

``` bash
python stress_rest_pipleine_v3.py \
  --root "data/Wearable_Dataset" \
  --output "results/paper_baseline"
```

Windows PowerShell equivalent:

``` powershell
python .\stress_rest_pipleine_v3.py --root ".\data\Wearable_Dataset" --output ".\results\paper_baseline"
```

Optional arguments include `--seed`, `--folds`, and
`--include-excluded`. The latter includes subjects that the
paper-aligned configuration excludes; do not enable it when aiming to
follow the paper's exclusion setup.

### CNN feature learning and classical ML

Run the subject-mixed experiment:

``` bash
python cnn_stress_contribution_v2.py \
  --root "data/Wearable_Dataset" \
  --output "results/cnn_mixed10" \
  --experiment mixed10
```

Run leave-one-subject-out evaluation:

``` bash
python cnn_stress_contribution_v2.py \
  --root "data/Wearable_Dataset" \
  --output "results/cnn_loso" \
  --experiment loso
```

Run both protocol-transfer directions:

``` bash
python cnn_stress_contribution_v2.py \
  --root "data/Wearable_Dataset" \
  --output "results/cnn_v1_to_v2" \
  --experiment v1_to_v2
```

``` bash
python cnn_stress_contribution_v2.py \
  --root "data/Wearable_Dataset" \
  --output "results/cnn_v2_to_v1" \
  --experiment v2_to_v1
```

Run all supported experiments:

``` bash
python cnn_stress_contribution_v2.py \
  --root "data/Wearable_Dataset" \
  --output "results/cnn_all" \
  --experiment all
```

For a quick pipeline/debug run, use `--max-splits 2`. This is a partial
run for debugging, not a final result. Common CNN options include
`--window-sec`, `--stride-sec`, `--epochs`, `--batch-size`,
`--embedding-dim`, `--lr`, `--seed`, and `--device auto|cpu|cuda`.

The focused `cnn_baseline_feature_comparison.py` script supports the
`mixed10` experiment and an optional handcrafted-feature CSV:

``` bash
python cnn_baseline_feature_comparison.py \
  --root "data/Wearable_Dataset" \
  --output "results/feature_comparison" \
  --experiment mixed10 \
  --handcrafted-csv "results/paper_baseline/stress_rest_features_intra_subject_normalized.csv"
```

Use `--help` to confirm the options available in your checked-out script
version:

``` bash
python stress_rest_pipleine_v3.py --help
python cnn_stress_contribution_v2.py --help
python cnn_baseline_feature_comparison.py --help
```

## Outputs

Output filenames depend on the selected script and experiment. The
scripts save CSV/JSON artifacts such as:

-   model metrics (Accuracy, Precision, Recall, and F1);
-   confusion matrices;
-   processed feature tables and missingness reports;
-   block manifests and window-build failure reports;
-   per-split train/test embeddings and predictions;
-   run metadata and configuration.

Treat generated results as experimental outputs. Record the exact
dataset version, script, command, random seed, split protocol, and
software versions when reporting results.

## Dataset limitations and quality checks

The dataset maintainers describe several session-specific issues,
including duplicated data for stress participant `S02`, invalid signals
for `f07` due to the wristband protection cover, and split/incomplete
recordings for some exercise sessions. The paper-aligned stress baseline
excludes `f07` and `f13` according to its implementation notes. Review
the dataset's `data_constraints.txt` and the original paper before
changing exclusions or interpreting subject-level results.

The CNN scripts also perform a segmentation audit and stop if the
generated block count does not match the expected 256 blocks. Do not
bypass that check without investigating the segmentation manifest and
failures.

## Reproducibility and responsible reporting

-   Fix and report the random seed.
-   Report whether evaluation is subject-mixed, LOSO, or cross-version
    transfer.
-   Fit resampling and learned transformations only on training data
    within each split.
-   Do not report debug/partial runs as final results.
-   Do not claim the published paper's accuracy unless your own run
    produces it under a comparable protocol.
-   Report deviations from the paper, including signal-processing
    choices that the paper does not specify.
-   The dataset contains physiological recordings from human
    participants. Use it in accordance with the dataset documentation,
    license, and research ethics requirements.

## Citation and license

If you use this dataset, cite both the dataset record and the original
article:

> Hongn, A., Bosch, F., Prado, L., & Bonomini, P. (2025). *Wearable
> Device Dataset from Induced Stress and Structured Exercise Sessions*
> (version 1.0.1). PhysioNet. https://doi.org/10.13026/he0v-tf17

> Hongn, A., Bosch, F., Prado, L. E., Ferrández, J. M., & Bonomini, M.
> P. (2025). *Wearable Physiological Signals under Acute Stress and
> Exercise Conditions*. Scientific Data, 12(1).
> https://doi.org/10.1038/s41597-025-04845-9

The dataset is distributed under the [Open Data Commons Attribution
License v1.0
(ODC-By)](https://physionet.org/content/wearable-device-dataset/view-license/1.0.1/).
Review the license and include the required attribution if
redistributing the database or derived database.

## License for this code

No code license is specified in this repository template. Add a
`LICENSE` file before publishing if you want to grant others explicit
reuse rights. The dataset license does not automatically determine the
license for your code.
