#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimum-Variance-Analysis of the “Rivera” switchback (PSP, 25 Feb 2022)

Author : <your-name-here>
Created: 2025-06-13
Licence: CC-BY-4.0

---------------------------------------------------------------------------
DEPENDENCIES
  pip install --upgrade pyspedas pytplot numpy scipy matplotlib

RUNTIME OPTIONS (edit if needed)
  - event centre UTC            : EVENT_CENTRE
  - half-window length (s)      : HALF_WIN
  - download cache directory    : SPEDAS_DATA_DIR
---------------------------------------------------------------------------

The code follows Sonnerup & Cahill (1967) / Sonnerup & Scheible (1998):

  • plain MVA  → eigenpair (λ₁ ≥ λ₂ ≥ λ₃, e₁,e₂,e₃);  n = e₃  
  • MVAB₀      → same on field components ⟂ ⟨B⟩  
  • reliability → λ₂/λ₃,   Δθ ≈ λ₃ / [ (N−1)(λ₂−λ₃) ]  [rad]

All outputs are saved under ./output_rivera_mva/.
"""

# --------------------------------------------------------------------- #
# 1. Imports & parameters
# --------------------------------------------------------------------- #
import os, pathlib, argparse
from datetime import datetime

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

import pyspedas
from pyspedas import get_data

# ---------- user-editable “knobs” ------------------------------------ #
EVENT_CENTRE = "2022-02-25T15:50:00Z"   # Rivera jump barycentre (UTC)
HALF_WIN     = 60                       # seconds each side of centre
TRANGE_FULL  = ["2022-02-25/15:00", "2022-02-25/16:40"]

# where PySPEDAS stores CDFs (create once, gets reused)
os.environ.setdefault("SPEDAS_DATA_DIR", str(pathlib.Path.home()/".pyspedas"))
# --------------------------------------------------------------------- #


def iso2unix(timestr: str) -> float:
    """Return Unix epoch seconds for an ISO-8601 UTC time string."""
    return datetime.fromisoformat(timestr.replace("Z", "+00:00")).timestamp()


def fetch_psp_mag(trange):
    """Download PSP FIELDS L2 MAG-RTN data and return (t,B) numpy arrays."""
    # PySPEDAS download / load
    mag_vars = pyspedas.projects.psp.fields(
        trange=trange,
        datatype="mag_rtn",
        level="l2",
        time_clip=True,
        no_update=False,   # redownload if remote file is newer
    )
    # get_data → pytplot variable object; item 0 = times, 1 = data
    t, b_rtn = get_data(mag_vars[0])
    return t, b_rtn.astype(np.float64)   # ensure float64 for precision


def mva(b_xyz: np.ndarray):
    """
    Classical minimum-variance analysis on array shape (N,3).

    Returns:
        lam  : eigenvalues sorted descending
        vecs : corresponding eigenvectors (columns) in RTN basis
    """
    npts = b_xyz.shape[0]
    b_demean = b_xyz - b_xyz.mean(axis=0, keepdims=True)
    cov = b_demean.T @ b_demean / npts
    lam, vecs = la.eigh(cov)       # ascending order
    order = lam.argsort()[::-1]    # → descending
    return lam[order], vecs[:, order]


def mvab0(b_xyz: np.ndarray):
    """MVAB₀ = MVA on components perpendicular to mean field."""
    b0 = b_xyz.mean(axis=0)
    b0_hat = b0 / np.linalg.norm(b0)
    # projection matrix onto ⟂ plane   P = I − b0_hat ⊗ b0_hat
    b_perp = b_xyz - (b_xyz @ b0_hat)[:, None] * b0_hat[None, :]
    return mva(b_perp)


def sonnerup_error(lam, npts) -> float:
    """Angular uncertainty Δθ (deg) from Sonnerup & Scheible (1998)."""
    lam1, lam2, lam3 = lam        # already sorted descending
    dtheta_rad = lam3 / ((npts - 1) * (lam2 - lam3))
    return np.degrees(dtheta_rad)


def report(title, lam, vecs, npts):
    lam1, lam2, lam3 = lam
    ratio23 = lam2 / lam3
    dtheta = sonnerup_error(lam, npts)
    nvec = vecs[:, 2]             # smallest variance direction

    print(f"\n——— {title} ———")
    print(f"λ₁, λ₂, λ₃  = {lam1:.3e}, {lam2:.3e}, {lam3:.3e}  (nT²)")
    print(f"λ₂/λ₃       = {ratio23:.2f}   {'⚠️  <2  ⇒  ill-defined' if ratio23 < 2 else ''}")
    print(f"n̂ (RTN)     = [{nvec[0]:+.3f}, {nvec[1]:+.3f}, {nvec[2]:+.3f}]")
    print(f"Δθ (1σ)     ≈ {dtheta:.2f}°")
    return nvec, ratio23, dtheta


def main():
    # ---------------------------------------------------------------- #
    # 2. Data acquisition
    # ---------------------------------------------------------------- #
    print("> Downloading PSP MAG-RTN Level-2 …")
    t_full, b_full = fetch_psp_mag(TRANGE_FULL)
    t0_unix = iso2unix(EVENT_CENTRE)
    mask_win = np.abs(t_full - t0_unix) <= HALF_WIN
    if mask_win.sum() < 10:
        raise RuntimeError("Window too small or wrong centre—got <10 samples.")
    t_win, b_win = t_full[mask_win], b_full[mask_win]

    print(f"  • full interval  : {len(t_full)} points")
    print(f"  • MVA window     : {len(t_win)} points  "
          f"({t_win[0]-t0_unix:+.1f}s  →  {t_win[-1]-t0_unix:+.1f}s)")

    # ---------------------------------------------------------------- #
    # 3. Perform analyses
    # ---------------------------------------------------------------- #
    lam_mva, vecs_mva = mva(b_win)
    lam_b0,  vecs_b0  = mvab0(b_win)

    # ---------------------------------------------------------------- #
    # 4. Numerical report
    # ---------------------------------------------------------------- #
    n_mva, r23_mva, dθ_mva = report("Plain MVA", lam_mva, vecs_mva, len(t_win))
    n_b0,  r23_b0,  dθ_b0  = report("MVAB₀",   lam_b0,  vecs_b0,  len(t_win))

    Δφ = np.degrees(np.arccos(np.clip(np.abs(np.dot(n_mva, n_b0)), 0, 1)))
    print(f"\nAngle between n̂_MVA and n̂_B0  :  {Δφ:.2f}°")

    # ---------------------------------------------------------------- #
    # 5. Plots
    # ---------------------------------------------------------------- #
    outdir = pathlib.Path("output_rivera_mva")
    outdir.mkdir(exist_ok=True)

    # 5.1 B components in RTN
    ts_rel = (t_win - t0_unix)
    plt.figure(figsize=(9, 5))
    plt.plot(ts_rel, b_win[:, 0], label="B_R")
    plt.plot(ts_rel, b_win[:, 1], label="B_T")
    plt.plot(ts_rel, b_win[:, 2], label="B_N")
    plt.axvline(0, color="k", lw=0.8, ls="--")
    plt.xlabel("t – t₀  [s]");  plt.ylabel("B  [nT]")
    plt.title("PSP MAG-RTN around Rivera jump")
    plt.legend();  plt.tight_layout()
    plt.savefig(outdir/"B_RTN.png", dpi=180)

    # 5.2 Field in LMN frame (plain MVA)
    L, M, N = vecs_mva.T          # columns are eigenvectors
    B_lmn = b_win @ np.vstack([L, M, N]).T
    plt.figure(figsize=(9, 5))
    plt.plot(ts_rel, B_lmn[:, 0], label="B_L (max)")
    plt.plot(ts_rel, B_lmn[:, 1], label="B_M (int)")
    plt.plot(ts_rel, B_lmn[:, 2], label="B_N (min)")
    plt.axhline(0, color="k", lw=0.6)
    plt.axvline(0, color="k", lw=0.8, ls="--")
    plt.xlabel("t – t₀  [s]");  plt.ylabel("B  [nT]")
    plt.title("Field in MVA frame (LMN)")
    plt.legend();  plt.tight_layout()
    plt.savefig(outdir/"B_LMN.png", dpi=180)

    # ---------------------------------------------------------------- #
    # 6. Save numbers to disk
    # ---------------------------------------------------------------- #
    np.savez(
        outdir/"mva_results.npz",
        n_mva=n_mva,  n_b0=n_b0,
        lam_mva=lam_mva, lam_b0=lam_b0,
        t=t_win,  B_RTN=b_win,  B_LMN=B_lmn
    )
    print(f"\nAll figures & data written to → {outdir.resolve()}")


# --------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
