from __future__ import annotations

import wave
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
ASSET_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = ASSET_DIR / "case_timbre_drift.svg"

DRIFT_SOURCE = ROOT / "evals" / "outputs" / "flac" / "reconstructions" / "1069-133709-0005.wav"
DRIFT_RECON = ROOT / "evals" / "outputs" / "neural-2k" / "reconstructions" / "1069-133709-0005.wav"
STABLE_SOURCE = ROOT / "evals" / "outputs" / "flac" / "reconstructions" / "1069-133709-0006.wav"
STABLE_RECON = ROOT / "evals" / "outputs" / "neural-2k" / "reconstructions" / "1069-133709-0006.wav"

N_FFT = 512
HOP = 128

ROWS = [
    ("Drift sample 0005, 2–3 s", DRIFT_SOURCE, DRIFT_RECON, 2.0, 3.0),
    ("Drift sample 0005, 6–7 s", DRIFT_SOURCE, DRIFT_RECON, 6.0, 7.0),
    ("Stable control 0006, 6–7 s", STABLE_SOURCE, STABLE_RECON, 6.0, 7.0),
]


def load_wav(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as handle:
        sample_rate = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())
    if channels != 1:
        raise ValueError(f"Expected mono wav: {path}")
    if width != 2:
        raise ValueError(f"Expected 16-bit PCM wav: {path}")
    waveform = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return sample_rate, waveform


def crop(waveform: np.ndarray, sample_rate: int, start_s: float, end_s: float) -> np.ndarray:
    start = int(start_s * sample_rate)
    end = int(end_s * sample_rate)
    return waveform[start:end]


def stft_mag(waveform: np.ndarray) -> np.ndarray:
    if waveform.size < N_FFT:
        waveform = np.pad(waveform, (0, N_FFT - waveform.size))
    pad = (HOP - (waveform.size - N_FFT) % HOP) % HOP
    if pad:
        waveform = np.pad(waveform, (0, pad))
    frames = 1 + (waveform.size - N_FFT) // HOP
    window = np.hanning(N_FFT).astype(np.float32)
    stacked = np.stack([waveform[i * HOP : i * HOP + N_FFT] * window for i in range(frames)], axis=0)
    spectrum = np.fft.rfft(stacked, axis=1)
    return np.abs(spectrum).T


def spec_db(magnitude: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(magnitude, 1e-5))


def metrics(src: np.ndarray, rec: np.ndarray) -> tuple[float, float]:
    corr = float(np.corrcoef(src, rec)[0, 1])
    src_mag = stft_mag(src)
    rec_mag = stft_mag(rec)
    log_diff = np.log10(np.maximum(src_mag, 1e-5)) - np.log10(np.maximum(rec_mag, 1e-5))
    lsd = float(np.sqrt(np.mean(log_diff**2)))
    return corr, lsd


def extent(sample_rate: int, n_frames: int) -> list[float]:
    duration = ((n_frames - 1) * HOP + N_FFT) / sample_rate
    return [0.0, duration, 0.0, sample_rate / 2000.0]


def main() -> None:
    plt.style.use("seaborn-v0_8-white")
    fig, axes = plt.subplots(len(ROWS), 3, figsize=(7.2, 7.9), constrained_layout=True)

    for row_idx, (label, src_path, rec_path, start_s, end_s) in enumerate(ROWS):
        sample_rate_src, source = load_wav(src_path)
        sample_rate_rec, recon = load_wav(rec_path)
        if sample_rate_src != sample_rate_rec:
            raise ValueError(f"Sample-rate mismatch: {src_path} vs {rec_path}")

        source = crop(source, sample_rate_src, start_s, end_s)
        recon = crop(recon, sample_rate_rec, start_s, end_s)

        source_mag = stft_mag(source)
        recon_mag = stft_mag(recon)
        source_db = spec_db(source_mag)
        recon_db = spec_db(recon_mag)
        diff = np.abs(np.log10(np.maximum(source_mag, 1e-5)) - np.log10(np.maximum(recon_mag, 1e-5)))

        corr, lsd = metrics(source, recon)
        e = extent(sample_rate_src, source_mag.shape[1])
        vmin = min(float(source_db.min()), float(recon_db.min()))
        vmax = max(float(source_db.max()), float(recon_db.max()))

        axes[row_idx, 0].imshow(source_db, origin="lower", aspect="auto", extent=e, cmap="magma", vmin=vmin, vmax=vmax)
        axes[row_idx, 1].imshow(recon_db, origin="lower", aspect="auto", extent=e, cmap="magma", vmin=vmin, vmax=vmax)
        axes[row_idx, 2].imshow(diff, origin="lower", aspect="auto", extent=e, cmap="viridis")

        axes[row_idx, 0].set_title(f"{label}\nsource", fontsize=9.5)
        axes[row_idx, 1].set_title(f"neural-2k\ncorr={corr:.3f}, local LSD={lsd:.3f}", fontsize=9.5)
        axes[row_idx, 2].set_title("absolute log-spectral difference", fontsize=9.5)

        for col_idx in range(3):
            ax = axes[row_idx, col_idx]
            ax.set_ylabel("kHz")
            ax.set_ylim(0, 8)
            ax.set_xlabel("Local time (s)")
            ax.grid(False)

    fig.suptitle("Local-window spectrogram analysis for neural-2k timbre drift", fontsize=11, y=1.01)
    fig.savefig(OUTPUT_PATH, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
