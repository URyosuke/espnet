#!/usr/bin/env python3
"""Compute D0 (mean Δ‑cepstrum norm) feature for ESPnet‑TTS datasets.

Usage: compute_d0.py --wavscp wav.scp --segments segments --outdir out
  * `wav.scp`   : utt_id wav_path
  * `segments`  : (optional) Kaldi segments file
  * `outdir`    : 出力先 (utt_id.npy を配置)
依存: numpy, scipy, soundfile, tqdm
"""
import argparse
import math
import os
from pathlib import Path
import soundfile as sf
import numpy as np
from scipy.fft import rfft, irfft
from tqdm import tqdm

# ---------------- cepstrum helpers -----------------

def spec2ceps(spec: np.ndarray, order: int) -> np.ndarray:
    """Power spectrum -> cepstrum (real). spec shape (F,)"""
    log_spec = np.log(spec + 1e-12)
    ceps = irfft(log_spec)[: order + 1]
    return ceps

def ceps2dCeps(ceps: np.ndarray, K: int = 1) -> np.ndarray:
    """Δ‑cepstrum along time axis (simple K‑frame diff)."""
    return np.diff(ceps, n=K, axis=0)

def dCeps2norm(dceps: np.ndarray, power: int = 1) -> np.ndarray:
    norm = np.linalg.norm(dceps, ord=2, axis=1) ** power
    return norm

# ---------------- main pipeline --------------------

FRAME_LEN = 400  # 25 ms @16 kHz
FRAME_SHIFT = 160  # 10 ms
N_FFT = 512
CEP_ORDER = 24


def framesig(sig: np.ndarray, frame_len: int, frame_shift: int):
    num_frames = 1 + (len(sig) - frame_len) // frame_shift
    idx = (
        np.tile(np.arange(0, frame_len), (num_frames, 1))
        + np.tile(np.arange(0, num_frames * frame_shift, frame_shift), (frame_len, 1)).T
    )
    return sig[idx]


def calc_d0(wav: np.ndarray, sr: int) -> np.ndarray:
    assert sr == 16000, "sample rate must be 16 kHz"
    frames = framesig(wav, FRAME_LEN, FRAME_SHIFT) * np.hamming(FRAME_LEN)
    spec = np.abs(rfft(frames, n=N_FFT)) ** 2  # power spec
    ceps = np.stack([spec2ceps(s, CEP_ORDER) for s in spec])
    dceps = ceps2dCeps(ceps, 1)
    norm = dCeps2norm(dceps, 1)
    return norm  # length T-1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wavscp", required=True)
    p.add_argument("--segments")
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    with open(args.wavscp) as f:
        wav_list = [line.strip().split() for line in f]

    for utt_id, wav_path in tqdm(wav_list):
        wav, sr = sf.read(wav_path)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        d0 = calc_d0(wav, sr)
        np.save(Path(args.outdir) / f"{utt_id}.npy", d0.astype(np.float32))

if __name__ == "__main__":
    main()