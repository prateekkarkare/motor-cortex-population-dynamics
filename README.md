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

> **Timing correction (2026-09-13):** the native loader returned nonchronological
> timestamps. The old workaround binned consecutive rows and then replaced their
> clock, misaligning spikes and trials. `bin_spikes` now sorts and validates the
> native clock before binning, and restores frequency metadata only when the
> timestamp values match exactly. Reload the raw data before using this fix.

## Status

**Verified split-half experiment.** Notebook section 6b uses all 137 held-in units
and 2,295 trials. Trials are split within 34 target directions, with neuron order
and normalization learned from half A only. Half B retains a clear diagonal;
median per-neuron A/B tuning correlation is **0.830**. Shuffling B's trial labels
removes the ridge. This supports reproducible direction-associated rate patterns,
not a proven cosine law or causal direction code.

![Corrected split-half heatmaps](assets/mcmaze_split_half_tuning.png)

**Preparation versus movement (2026-09-14).** Section 8 adds event-aligned neural
and kinematic plots plus same-neuron condition-tuning comparisons, using exact
raw-spike windows on 1,967 long-delay trials and 108 maze/version conditions.
Preparation/movement correlation has median **0.050** across all units, but
preparatory reliability is often low. For 48 units passing an exploratory
within-epoch reliability screen in both epochs, the cross-epoch median is
**0.178**. See the [aligned traces](assets/mcmaze_preparation_movement_traces.png)
and [reliability controls](assets/mcmaze_preparation_movement_correlations.png).
These are paper-inspired analyses, not proof of a dynamical mechanism.

**Historical results need rerunning.** The earlier cosine-fit and static-PCA
numbers used the faulty binning. They remain in the experiment log as history,
but must not be treated as current findings. The 2010 paper's PCA also uses
conditions x (neurons x time), not our static direction x neuron matrix; its
Figure 8 oscillator is a simulation, not an observed ring in these data.

**Verification.** Synthetic split-half controls and a shuffled-timestamp binning
regression passed. Corrected rates agree with raw NWB spike timestamps for
25 sampled trial/neuron pairs; this is a spot check, not a full-data audit.

**Next.** Recompute the earlier tuning/PCA analyses on the corrected clock, then
develop a population decoder with held-out whole trials.

Full experiment log: [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) · Figures: [`assets/`](assets/)
