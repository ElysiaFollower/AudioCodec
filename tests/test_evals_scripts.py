from __future__ import annotations

import tempfile
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import unittest
from unittest import mock

import torch


EVALS_SCRIPTS = Path(__file__).resolve().parents[1] / "evals" / "scripts"
if str(EVALS_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(EVALS_SCRIPTS))

from _common import (
    bytes_to_kbps,
    derive_item_id,
    pcm16_bytes,
    read_jsonl,
    rvq_payload_bytes,
    save_audio,
    write_jsonl,
)
from _metrics import compute_log_spectral_distance, compute_multi_scale_stft, compute_si_sdr_db
from diagnose_representations import (
    DIAGNOSTICS_SCHEMA_VERSION,
    _scope_specs,
    run_diagnostics,
)
from export_neural_codec import (
    CONTEXT_EXPORT_SCHEMA_VERSION,
    _build_manifest_row,
    _nominal_bitrate_kbps,
    _resolve_context_window_frames,
    _resolve_context_window_seconds,
    _rvq_payload_bits,
    main as export_neural_codec_main,
)
from audiocodec.config import load_experiment_config
from audiocodec.models.codec import build_codec_model


class EvalHelpersTest(unittest.TestCase):
    def test_pcm16_bytes(self) -> None:
        self.assertEqual(pcm16_bytes(16000, channels=1), 32000)

    def test_bytes_to_kbps(self) -> None:
        self.assertAlmostEqual(bytes_to_kbps(3000, 2.0), 12.0)

    def test_rvq_payload_bytes(self) -> None:
        self.assertEqual(rvq_payload_bytes(num_frames=100, num_quantizers=24, codebook_size=1024), 3000)

    def test_derive_item_id_uses_relative_path(self) -> None:
        root = Path("/tmp/librispeech")
        path = root / "speaker" / "chapter" / "utt.flac"
        self.assertEqual(derive_item_id(path, dataset_root=root), "speaker__chapter__utt")


class NeuralExportMetadataTest(unittest.TestCase):
    def test_context_window_resolution_uses_frame_rate(self) -> None:
        self.assertEqual(
            _resolve_context_window_frames(
                context_window_seconds=2.5,
                context_window_frames=None,
                frame_rate=50,
            ),
            125,
        )
        self.assertEqual(
            _resolve_context_window_frames(
                context_window_seconds=2.5,
                context_window_frames=64,
                frame_rate=50,
            ),
            64,
        )
        self.assertAlmostEqual(
            _resolve_context_window_seconds(
                context_window_seconds=None,
                context_window_frames=125,
                frame_rate=50,
            ),
            2.5,
        )

    def test_context_window_rejects_negative_values(self) -> None:
        with self.assertRaises(ValueError):
            _resolve_context_window_frames(
                context_window_seconds=-1.0,
                context_window_frames=None,
                frame_rate=50,
            )
        with self.assertRaises(ValueError):
            _resolve_context_window_seconds(
                context_window_seconds=None,
                context_window_frames=-1,
                frame_rate=50,
            )

    def test_build_manifest_row_records_phase_one_schema(self) -> None:
        row = _build_manifest_row(
            source_row={"id": "utt", "pcm16_bytes": 64_000},
            source_path=Path("/tmp/source.wav"),
            reconstruction_path=Path("/tmp/out/reconstructions/utt.wav"),
            codes_path=Path("/tmp/out/representations/utt.codes.pt"),
            latent_path=Path("/tmp/out/representations/utt.latent.pt"),
            quantized_path=Path("/tmp/out/representations/utt.quantized.pt"),
            duration_seconds=2.0,
            num_samples=32_000,
            sample_rate=16_000,
            channels=1,
            num_frames=100,
            frame_rate=50,
            hop_length=320,
            latent_dim=128,
            num_quantizers=4,
            codebook_size=1024,
            bits_per_code=10,
            payload_bytes=500,
            codec_label="neural-4k",
            checkpoint_path=Path("/tmp/checkpoint.pt"),
            checkpoint_step=7,
            config_path=Path("/tmp/config.json"),
            context_scope="long",
            context_window_seconds=10.0,
            context_window_frames=500,
            clip_scope="full_utterance",
            is_full_utterance=True,
        )

        self.assertEqual(row["schema_version"], CONTEXT_EXPORT_SCHEMA_VERSION)
        self.assertEqual(row["latent_path"], "/tmp/out/representations/utt.latent.pt")
        self.assertEqual(row["quantized_path"], "/tmp/out/representations/utt.quantized.pt")
        self.assertEqual(row["frame_rate"], 50)
        self.assertEqual(row["hop_length"], 320)
        self.assertEqual(row["latent_dim"], 128)
        self.assertEqual(row["rvq_payload_bits"], _rvq_payload_bits(100, 4, 10))
        self.assertAlmostEqual(row["nominal_bitrate_kbps"], _nominal_bitrate_kbps(50, 4, 10))
        self.assertEqual(row["context_scope"], "long")
        self.assertEqual(row["context_window_seconds"], 10.0)
        self.assertEqual(row["context_window_frames"], 500)
        self.assertTrue(row["is_full_utterance"])

    def test_export_neural_codec_saves_phase_one_representations(self) -> None:
        config = load_experiment_config("configs/baseline.json")
        model = build_codec_model(config)

        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            checkpoint_path = root / "checkpoint.pt"
            source_path = root / "source.wav"
            manifest_path = root / "manifest.jsonl"
            output_dir = root / "output"
            waveform = torch.randn(config.audio.channels, 1_600) * 0.01

            save_audio(source_path, waveform, config.audio.sample_rate)
            torch.save({"model": model.state_dict(), "config": config.to_dict(), "step": 3}, checkpoint_path)
            write_jsonl(
                manifest_path,
                [
                    {
                        "id": "utt",
                        "source_path": str(source_path),
                        "duration_seconds": waveform.shape[-1] / config.audio.sample_rate,
                        "num_samples": waveform.shape[-1],
                        "sample_rate": config.audio.sample_rate,
                        "channels": config.audio.channels,
                        "pcm16_bytes": pcm16_bytes(waveform.shape[-1], channels=config.audio.channels),
                    }
                ],
            )

            argv = [
                "export_neural_codec.py",
                "--manifest",
                str(manifest_path),
                "--checkpoint",
                str(checkpoint_path),
                "--output-dir",
                str(output_dir),
                "--codec-label",
                "test-neural",
                "--device",
                "cpu",
                "--save-representations",
                "--context-scope",
                "full_utterance",
                "--is-full-utterance",
            ]
            with mock.patch.object(sys, "argv", argv):
                with redirect_stdout(io.StringIO()):
                    export_neural_codec_main()

            rows = read_jsonl(output_dir / "manifest.jsonl")
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["schema_version"], CONTEXT_EXPORT_SCHEMA_VERSION)
            self.assertEqual(row["context_scope"], "full_utterance")
            self.assertTrue(row["is_full_utterance"])
            self.assertTrue(Path(row["reconstruction_path"]).exists())
            self.assertTrue(Path(row["codes_path"]).exists())
            self.assertTrue(Path(row["latent_path"]).exists())
            self.assertTrue(Path(row["quantized_path"]).exists())

            codes = torch.load(row["codes_path"], map_location="cpu")
            latent = torch.load(row["latent_path"], map_location="cpu")
            quantized = torch.load(row["quantized_path"], map_location="cpu")
            self.assertEqual(codes.shape[:2], (1, config.quantizer.num_quantizers))
            self.assertEqual(latent.shape, quantized.shape)
            self.assertEqual(latent.shape[0], 1)
            self.assertEqual(latent.shape[1], config.model.latent_dim)


class RepresentationDiagnosticsTest(unittest.TestCase):
    def test_run_diagnostics_writes_rows_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            export_dir = root / "export"
            output_dir = export_dir / "diagnostics"
            reps_dir = export_dir / "representations"
            reps_dir.mkdir(parents=True)

            latent = torch.arange(12, dtype=torch.float32).view(1, 2, 6)
            quantized = latent + 1.0
            codes = torch.tensor([[[1, 1, 2, 2, 2, 3], [4, 4, 4, 5, 5, 5]]], dtype=torch.long)
            latent_path = reps_dir / "utt.latent.pt"
            quantized_path = reps_dir / "utt.quantized.pt"
            codes_path = reps_dir / "utt.codes.pt"
            torch.save(latent, latent_path)
            torch.save(quantized, quantized_path)
            torch.save(codes, codes_path)

            manifest_path = export_dir / "manifest.jsonl"
            write_jsonl(
                manifest_path,
                [
                    {
                        "id": "utt",
                        "frame_rate": 2,
                        "latent_path": str(latent_path),
                        "quantized_path": str(quantized_path),
                        "codes_path": str(codes_path),
                    }
                ],
            )
            args = type(
                "Args",
                (),
                {
                    "local_window_seconds": 0.5,
                    "medium_window_seconds": 1.5,
                    "long_window_seconds": 3.0,
                },
            )()

            diagnostics, summary = run_diagnostics(
                export_dir=export_dir,
                output_dir=output_dir,
                manifest_path=manifest_path,
                representations=["latent", "quantized", "codes"],
                scope_specs=_scope_specs(args),
            )

            self.assertEqual(len(diagnostics), 12)
            self.assertTrue((output_dir / "diagnostics.jsonl").exists())
            self.assertTrue((output_dir / "summary.json").exists())
            self.assertEqual(summary["schema_version"], DIAGNOSTICS_SCHEMA_VERSION)
            self.assertEqual(len(summary["gate_recommendations"]), 3)
            latent_local = [
                row
                for row in diagnostics
                if row["representation"] == "latent" and row["context_scope"] == "local"
            ][0]
            self.assertEqual(latent_local["schema_version"], DIAGNOSTICS_SCHEMA_VERSION)
            self.assertEqual(latent_local["metric_family"], "continuous_past_window_mean")
            self.assertEqual(latent_local["context_window_frames"], 1)
            self.assertIsNotNone(latent_local["normalized_mse"])
            code_full = [
                row
                for row in diagnostics
                if row["representation"] == "codes" and row["context_scope"] == "full_utterance"
            ][0]
            self.assertEqual(code_full["metric_family"], "code_window_reuse")
            self.assertIsNotNone(code_full["window_reuse_rate"])
            self.assertIsNotNone(code_full["marginal_entropy_bits_per_code"])


class EvalMetricsTest(unittest.TestCase):
    def test_identical_waveforms_have_near_zero_distance(self) -> None:
        waveform = torch.randn(1, 16000)
        self.assertLess(compute_log_spectral_distance(waveform, waveform), 1e-4)
        self.assertLess(compute_multi_scale_stft(waveform, waveform), 1e-6)

    def test_identical_waveforms_have_high_si_sdr(self) -> None:
        waveform = torch.randn(1, 16000)
        self.assertGreater(compute_si_sdr_db(waveform, waveform), 60.0)


if __name__ == "__main__":
    unittest.main()
