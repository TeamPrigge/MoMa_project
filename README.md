# MoMa — CRN Cloud analysis notebooks (ASAP-030013)

Meta-analysis of ASAP CRN transcriptomic data: is tyrosinase (TYR) expression in
dopaminergic neurons, and dopamine catabolism (MAOA, MAOB, COMT, ALDH1A1),
associated with neuronal vulnerability in Parkinson's disease?

## Contents

| File | What it does |
|---|---|
| `01_tyr_mao_feasibility.ipynb` | Finds substantia nigra DA nuclei by markers, checks how the CRN pipeline labelled them, measures TYR/MAO detection, builds donor-level pseudobulk, first PD vs control look |
| `02_lowgene_simulation.ipynb` | Simulation study (synthetic data): detection vs depth, why HVG-filtered objects miss the panel, the QC/ambient trap, donor-level power for SN and LC capture scenarios, subtype contrasts, noise-amplification traps, GeoMx and bulk designs |
| `sim/lowgene_sim.py` | The simulation code behind notebook 02 (`python sim/lowgene_sim.py [--fast]`); writes `sim/figures/*.png` and `sim/sim_results.csv`. GeoMx section calibrated to the public Edwards/Chatterton TH-mask object |
| `moma_utils.py` | Helpers: GCS discovery, lazy `.h5ad` reading (only the genes/nuclei needed), CDE metadata join, marker-based DA calling, donor pseudobulk |
| `requirements.txt` | Python packages |

## Running in Verily Workbench

1. Push this folder to a GitHub repo (e.g. `moma-crn`).
2. In your workspace: **Apps → + Add Repository** → paste the repo URL.
3. Create a JupyterLab app (`n1-highmem-8` or larger; the SN subset is pulled lazily, but the
   `.h5ad` is first copied to the app disk — give it enough disk).
4. In the app terminal: `pip install -r ~/repos/moma-crn/requirements.txt`
5. Open `01_tyr_mao_feasibility.ipynb`, set `DATASET = "lee"`, run top to bottom.
   Section 2 is marked **CHECK**: confirm the region / donor / diagnosis / cell-type columns.
6. Repeat with `DATASET = "sulzer"`.

Outputs go to `~/moma_outputs/<dataset>/` and stay in the workspace (award §5).

## Developing outside Workbench (e.g. Colab)

Set `MOMA_TEST_H5AD=/path/to/any.h5ad` before starting the kernel. The notebook then skips
GCS and runs on that file, so you can develop on public or synthetic data. Never copy CRN data out.

## Status

Tested end-to-end on a synthetic AnnData shaped like the pipeline output.
Not yet run on CRN data — the bucket paths and obs column names must be confirmed on first run.
