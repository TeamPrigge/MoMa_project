"""
moma_utils.py — helpers for the MoMa meta-analysis (ASAP-030013) on CRN Cloud.

Question: is tyrosinase (TYR) expression in dopaminergic (DA) neurons, and the
dopamine-catabolism genes (MAOA, MAOB, COMT, ALDH1A1), associated with
vulnerability in Parkinson's disease?

Everything here is written to run INSIDE Verily Workbench on the curated
AnnData (.h5ad) outputs of the ASAP sc/snRNA-seq pipeline. Keep CRN data and
anything derived from it inside the workspace until it is publicly released.

Design notes
- Large .h5ad files are read lazily with h5py: obs/var are loaded, but counts
  are pulled only for the selected nuclei and the handful of genes we need.
  This avoids loading multi-GB matrices into memory.
- Raw UMI counts are taken from layers["counts"] (the pipeline stores raw
  counts there before normalisation). TYR is lowly expressed, so we work
  with UMIs and donor-level pseudobulk, never with log-normalised values.
- The donor (not the nucleus) is the unit of comparison.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

try:  # anndata >= 0.11
    from anndata.io import read_elem, sparse_dataset
except ImportError:  # older anndata
    from anndata.experimental import read_elem, sparse_dataset


# ---------------------------------------------------------------------------
# Gene panel
# ---------------------------------------------------------------------------

GENE_PANEL = {
    # neuromelanin synthesis (the MoMa hypothesis)
    "TYR": "tyrosinase",
    # dopaminergic identity
    "TH": "tyrosine hydroxylase",
    "SLC6A3": "dopamine transporter (DAT)",
    "SLC18A2": "VMAT2",
    "DDC": "aromatic L-amino acid decarboxylase",
    # noradrenergic marker, to flag LC-like / NE nuclei
    "DBH": "dopamine beta-hydroxylase",
    # dopamine catabolism
    "MAOA": "monoamine oxidase A",
    "MAOB": "monoamine oxidase B",
    "COMT": "catechol-O-methyltransferase",
    "ALDH1A1": "aldehyde dehydrogenase 1A1",
    # vulnerable vs resistant DA subtypes (Kamath et al. 2022)
    "SOX6": "SOX6 (ventral tier, vulnerable)",
    "AGTR1": "AGTR1 (most vulnerable SOX6 subtype)",
    "CALB1": "calbindin (dorsal tier, resistant)",
    # pan-neuronal, for the neuron denominator
    "RBFOX3": "NeuN",
    "SNAP25": "SNAP25",
}


# ---------------------------------------------------------------------------
# Finding and fetching files on GCS
# ---------------------------------------------------------------------------

def gcs_ls(prefix: str, recursive: bool = True) -> list[str]:
    """List objects under a gs:// prefix with gsutil (available in Workbench)."""
    if shutil.which("gsutil") is None:
        raise RuntimeError("gsutil not found — run this inside a Workbench app.")
    cmd = ["gsutil", "ls"] + (["-r"] if recursive else []) + [prefix.rstrip("/")]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"gsutil ls failed for {prefix}:\n{out.stderr.strip()}")
    return [l.strip() for l in out.stdout.splitlines() if l.strip() and not l.endswith(":")]


def find_files(paths: list[str], pattern: str) -> list[str]:
    """Filter a list of paths by a regex (e.g. r'final_adata\\.h5ad$')."""
    rx = re.compile(pattern)
    return sorted(p for p in paths if rx.search(p))


def pick_h5ad(paths):
    """Best curated AnnData in a listing: final_adata > merged_adata_object; never preprocess/."""
    h5ads = [p for p in paths if p.endswith(".h5ad") and "/preprocess/" not in p]
    for pat in (r"\.final\.h5ad$", r"final_adata\.h5ad$", r"merged_cleaned_unfiltered\.h5ad$", r"merged_adata_object\.h5ad$"):
        hits = find_files(h5ads, pat)
        if hits:
            return hits[0]
    return None


def gcs_size(path: str) -> str:
    """Human-readable size of a gs:// object or prefix."""
    out = subprocess.run(["gsutil", "du", "-sh", path], capture_output=True, text=True)
    parts = out.stdout.split()
    return " ".join(parts[:2]) if out.returncode == 0 and len(parts) >= 2 else "?"


def fetch(gs_path: str, local_dir: str | os.PathLike) -> Path:
    """Copy a gs:// object to the app's local disk (skip if already there)."""
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    dest = local_dir / gs_path.rstrip("/").split("/")[-1]
    if not dest.exists():
        subprocess.run(["gsutil", "-m", "cp", gs_path, str(dest)], check=True)
    return dest


# ---------------------------------------------------------------------------
# Lazy h5ad access
# ---------------------------------------------------------------------------

class LazyH5AD:
    """Minimal lazy reader: obs/var in memory, matrices sliced on demand."""

    def __init__(self, path: str | os.PathLike):
        self.path = str(path)
        with h5py.File(self.path, "r") as f:
            self.obs = read_elem(f["obs"])
            self.var = read_elem(f["var"])
            self.layers = list(f["layers"].keys()) if "layers" in f else []
            self.obs_keys_m = list(f["obsm"].keys()) if "obsm" in f else []
        self.n_obs, self.n_vars = len(self.obs), len(self.var)

    def __repr__(self):
        return (f"LazyH5AD({Path(self.path).name}: {self.n_obs:,} nuclei x {self.n_vars:,} genes; "
                f"layers={self.layers})")

    def _matrix_key(self, layer: str | None) -> str:
        if layer is None:
            return "X"
        if layer not in self.layers:
            import warnings
            warnings.warn(f"layer '{layer}' not in file (have: {self.layers}); using X. "
                          "Check that X holds raw counts (integers).")
            return "X"
        return f"layers/{layer}"

    def counts(self, rows: np.ndarray, cols: list[int], layer: str | None = "counts",
               chunk: int = 20_000) -> sp.csr_matrix:
        """Return rows x cols of the chosen matrix, reading rows in chunks."""
        rows = np.sort(np.asarray(rows))
        blocks = []
        with h5py.File(self.path, "r") as f:
            node = f[self._matrix_key(layer)]
            if isinstance(node, h5py.Dataset):          # dense matrix
                for i in range(0, len(rows), chunk):
                    r = rows[i:i + chunk]
                    blocks.append(sp.csr_matrix(node[r, :][:, cols]))
            else:                                       # sparse matrix
                ds = sparse_dataset(node)
                fmt = node.attrs.get("encoding-type", "csr_matrix")
                for i in range(0, len(rows), chunk):
                    r = rows[i:i + chunk]
                    m = ds[r] if fmt.startswith("csr") else ds[r, :]
                    blocks.append(sp.csr_matrix(m)[:, cols])
        return sp.vstack(blocks).tocsr() if blocks else sp.csr_matrix((0, len(cols)))

    def row_sums(self, rows: np.ndarray, layer: str | None = "counts", chunk: int = 20_000) -> np.ndarray:
        """Total UMIs per nucleus (fallback if obs has no total_counts)."""
        rows = np.sort(np.asarray(rows))
        out = []
        with h5py.File(self.path, "r") as f:
            node = f[self._matrix_key(layer)]
            ds = node if isinstance(node, h5py.Dataset) else sparse_dataset(node)
            for i in range(0, len(rows), chunk):
                m = ds[rows[i:i + chunk]]
                out.append(np.asarray(m.sum(axis=1)).ravel())
        return np.concatenate(out) if out else np.array([])


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def suggest_columns(df: pd.DataFrame, keywords: list[str]) -> list[str]:
    """Columns whose name contains any keyword (case-insensitive)."""
    kws = [k.lower() for k in keywords]
    return [c for c in df.columns if any(k in c.lower() for k in kws)]


def show_values(df: pd.DataFrame, cols: list[str], top: int = 12) -> None:
    for c in cols:
        vc = df[c].astype(str).value_counts()
        print(f"\n{c}  ({vc.size} distinct)")
        print(vc.head(top).to_string())


def resolve_genes(var: pd.DataFrame, symbols: list[str]) -> dict[str, int | None]:
    """Map gene symbols to column positions. Looks at var_names and common symbol columns."""
    candidates = [pd.Index(var.index.astype(str))]
    for col in ("gene_symbol", "gene_symbols", "feature_name", "gene_name", "gene_names", "symbol"):
        if col in var.columns:
            candidates.append(pd.Index(var[col].astype(str)))
    found = {}
    for s in symbols:
        pos = None
        for idx in candidates:
            hits = np.flatnonzero(idx.str.upper() == s.upper())
            if hits.size:
                pos = int(hits[0])
                break
        found[s] = pos
    return found


# ---------------------------------------------------------------------------
# Cell calling and quantification
# ---------------------------------------------------------------------------

def call_catecholaminergic(counts: pd.DataFrame, min_umi: int = 1) -> pd.Series:
    """
    Marker-based call, independent of the pipeline's cell-type labels.

    DA       : TH+ and (SLC6A3+ or SLC18A2+) and DBH-
    NE_like  : TH+ and DBH+   (LC-type; should be rare in SN)
    TH_only  : TH+ without DAT/VMAT2 (ambiguous, kept separate)
    other    : everything else
    """
    pos = counts >= min_umi
    th = pos.get("TH", False)
    dat_vmat = pos.get("SLC6A3", False) | pos.get("SLC18A2", False)
    dbh = pos.get("DBH", pd.Series(False, index=counts.index))
    call = pd.Series("other", index=counts.index, dtype=object)
    call[th & dat_vmat & ~dbh] = "DA"
    call[th & dbh] = "NE_like"
    call[th & ~dat_vmat & ~dbh] = "TH_only"
    return call.astype("category")


def donor_pseudobulk(counts: pd.DataFrame, total_umi: pd.Series, donor: pd.Series,
                     mask: pd.Series | None = None) -> pd.DataFrame:
    """
    Per donor: number of nuclei, summed UMIs, CPM of the pseudobulk, and
    fraction of nuclei with >=1 UMI, for every gene column in `counts`.
    """
    if mask is not None:
        counts, total_umi, donor = counts[mask], total_umi[mask], donor[mask]
    g = counts.groupby(donor, observed=True)
    umi = g.sum()
    det = (counts > 0).groupby(donor, observed=True).mean()
    tot = total_umi.groupby(donor, observed=True).sum()
    out = pd.DataFrame({"n_nuclei": g.size(), "total_umi": tot})
    for gene in counts.columns:
        out[f"{gene}_umi"] = umi[gene]
        out[f"{gene}_cpm"] = umi[gene] / tot * 1e6
        out[f"{gene}_frac_detected"] = det[gene]
    return out


def compare_groups(df: pd.DataFrame, value: str, group: str, a: str, b: str) -> dict:
    """Donor-level Mann-Whitney U test between two groups (exploratory)."""
    from scipy.stats import mannwhitneyu
    x = df.loc[df[group] == a, value].dropna()
    y = df.loc[df[group] == b, value].dropna()
    res = {"value": value, f"n_{a}": len(x), f"n_{b}": len(y),
           f"median_{a}": x.median(), f"median_{b}": y.median(), "p_mannwhitney": np.nan}
    if len(x) >= 3 and len(y) >= 3:
        res["p_mannwhitney"] = mannwhitneyu(x, y, alternative="two-sided").pvalue
    return res


def annotate_obs(obs, meta_dir):
    """Join brain region, subject and diagnosis from the CDE tables onto obs via the sample ID."""
    meta_dir = Path(meta_dir)
    def load(name):
        p = meta_dir / f"{name}.csv"
        return pd.read_csv(p) if p.exists() else None
    def lookup(df, key, col, by):
        if df is None or key not in df or col not in df:
            return None
        return by.map(df.drop_duplicates(key).set_index(key)[col])
    o = obs.copy()
    sample = o["sample"].astype(str).str.replace(r"_rep\d+$", "", regex=True)
    pm, sa, su, cp = (load(n) for n in ("PMDBS", "SAMPLE", "SUBJECT", "CLINPATH"))
    reg = lookup(pm, "ASAP_sample_id", "brain_region", sample)
    if reg is not None:
        o["brain_region"] = reg
    subj = lookup(sa, "ASAP_sample_id", "ASAP_subject_id", sample)
    if subj is not None:
        o["ASAP_subject_id"] = subj
        for df, cols in ((su, ("primary_diagnosis", "sex")),
                         (cp, ("age_at_death", "duration_pmi", "path_braak_asyn"))):
            for col in cols:
                v = lookup(df, "ASAP_subject_id", col, subj)
                if v is not None:
                    o[col] = v
    return o
