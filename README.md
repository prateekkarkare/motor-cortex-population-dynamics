# motor-cortex-population-dynamics

## Dataset

**MC_Maze** — delayed center-out / maze reaches recorded from macaque primary
motor cortex (M1) and dorsal premotor cortex (PMd) by the Churchland & Shenoy
labs. Distributed via the [Neural Latents Benchmark](https://neurallatents.github.io/)
and the DANDI archive as **dandiset 000128**.

Loaded here with [`nlb_tools`](https://github.com/neurallatents/nlb_tools)
(`NWBDataset`), which turns the NWB file into binned spike counts plus trial
info (reach conditions, hand position/velocity).

## Layout

```
notebooks/
  01_load_and_explore.ipynb   # load MC_Maze -> binned spikes + reach conditions
data/                          # DANDI download (gitignored — large)
requirements.txt
```

## Setup

`nlb_tools` pins `pandas <= 1.3.4`, whose only macOS wheels target Python **3.9**,
so build the env on 3.9 (macOS ships it at `/usr/bin/python3`).

```bash
/usr/bin/python3 -m venv .venv          # Python 3.9
source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps nlb_tools         # its pandas<=1.3.4 pin has no arm64 wheel;
                                        # runs fine against the pinned pandas 1.5.3
```

Download MC_Maze (dandiset 000128). The `dandi` CLI that still supports Python 3.9
is now rejected by the DANDI server (needs a newer client), so fetch the two NWB
assets straight from the DANDI API:

```bash
mkdir -p data/000128/sub-Jenkins
base=https://api.dandiarchive.org/api/dandisets/000128/versions/draft/assets
curl -L "$base/26e85f09-39b7-480f-b337-278a8f034007/download/" \
  -o data/000128/sub-Jenkins/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb   # ~690 MB
curl -L "$base/1bd112a4-5ec5-4033-ac30-d88e70e993d9/download/" \
  -o data/000128/sub-Jenkins/sub-Jenkins_ses-full_desc-test_ecephys.nwb             # ~3 MB
```

> **pandas compat:** `nlb_tools` 0.0.4 was written for pandas ≤ 1.3.4. Two calls
> (`resample`, `make_trial_data`) break on pandas ≥ 1.5; the notebook's `bin_spikes`
> works around both by rebuilding a clean regular time index after binning.

## Roadmap

1. **Load + explore** (this commit) — binned spikes, reach conditions, hand velocity.
2. Per-neuron directional tuning curves (firing rate vs reach direction).
3. PCA of the population — how many dimensions capture the reaching activity?
4. Reach-direction decoder.
