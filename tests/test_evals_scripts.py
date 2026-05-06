from __future__ import annotations

import tempfile
from contextlib import redirect_stdout
import io
from pathlib import Path
import subprocess
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
from collect_context_results import (
    CONTEXT_RESULTS_SCHEMA_VERSION,
    collect_context_results,
)
from evaluate_code_priors import (
    CODE_PRIOR_SCHEMA_VERSION,
    TOKEN_ORDERING,
    run_code_prior_evaluation,
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
from train_code_prior import train_code_prior
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


class CodePriorEntropyTest(unittest.TestCase):
    def test_run_code_prior_evaluation_writes_entropy_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            export_dir = root / "export"
            reps_dir = export_dir / "representations"
            reps_dir.mkdir(parents=True)

            codes = torch.tensor([[[0, 0, 0, 1, 1, 1], [2, 2, 2, 3, 3, 3]]], dtype=torch.long)
            codes_path = reps_dir / "utt.codes.pt"
            torch.save(codes, codes_path)
            manifest_path = export_dir / "manifest.jsonl"
            write_jsonl(
                manifest_path,
                [
                    {
                        "id": "utt",
                        "frame_rate": 2,
                        "num_quantizers": 2,
                        "codebook_size": 4,
                        "bits_per_code": 2,
                        "codes_path": str(codes_path),
                    }
                ],
            )

            rows, summary = run_code_prior_evaluation(
                export_dir=export_dir,
                manifest_path=manifest_path,
                eval_export_dir=None,
                eval_manifest_path=None,
                output_dir=export_dir / "code_priors",
                priors=["unigram", "previous_frame"],
                codebook_size=None,
                smoothing=1e-3,
            )

            self.assertEqual(len(rows), 4)
            self.assertTrue((export_dir / "code_priors" / "train_metrics.jsonl").exists())
            self.assertTrue((export_dir / "code_priors" / "val_metrics.jsonl").exists())
            self.assertTrue((export_dir / "code_priors" / "summary.json").exists())
            self.assertTrue((export_dir / "code_priors" / "config.json").exists())
            self.assertEqual(summary["schema_version"], CODE_PRIOR_SCHEMA_VERSION)
            self.assertEqual(summary["token_ordering"], TOKEN_ORDERING)
            self.assertEqual(summary["evaluation_split"], "self_eval")
            self.assertEqual(summary["num_quantizers"], 2)
            self.assertEqual(len(summary["priors"]), 2)
            unigram = [item for item in summary["priors"] if item["prior_family"] == "unigram"][0]
            previous = [item for item in summary["priors"] if item["prior_family"] == "previous_frame"][0]
            self.assertEqual(len(unigram["stage_bits_per_code"]), 2)
            self.assertLessEqual(
                previous["estimated_entropy_bitrate_kbps"],
                unigram["estimated_entropy_bitrate_kbps"],
            )
            self.assertIn("entropy_savings_ratio", previous)
            self.assertEqual(len(summary["blocked_priors"]), 3)
            val_rows = read_jsonl(export_dir / "code_priors" / "val_metrics.jsonl")
            self.assertEqual(val_rows[0]["schema_version"], CODE_PRIOR_SCHEMA_VERSION)
            self.assertEqual(val_rows[0]["dataset_split"], "self_eval")


class TrainCodePriorTest(unittest.TestCase):
    def _write_codes_export(self, root: Path) -> tuple[Path, Path]:
        export_dir = root / "export"
        reps_dir = export_dir / "representations"
        reps_dir.mkdir(parents=True)
        codes = torch.tensor(
            [
                [[0, 0, 1, 1, 2, 2, 3, 3], [1, 1, 1, 2, 2, 2, 3, 3]],
                [[3, 3, 2, 2, 1, 1, 0, 0], [2, 2, 2, 1, 1, 1, 0, 0]],
            ],
            dtype=torch.long,
        )
        codes_path = reps_dir / "utt.codes.pt"
        torch.save(codes, codes_path)
        manifest_path = export_dir / "manifest.jsonl"
        write_jsonl(
            manifest_path,
            [
                {
                    "id": "utt",
                    "frame_rate": 2,
                    "num_quantizers": 2,
                    "codebook_size": 4,
                    "bits_per_code": 2,
                    "codes_path": str(codes_path),
                }
            ],
        )
        return export_dir, manifest_path

    def _args(self, prior: str) -> object:
        return type(
            "Args",
            (),
            {
                "prior": prior,
                "codebook_size": None,
                "max_train_items": None,
                "max_eval_items": None,
                "steps": 2 if prior == "local_tcn" else 1,
                "batch_size": 2,
                "sequence_length": 6,
                "eval_every": 1,
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
                "embedding_dim": 8,
                "hidden_dim": 8,
                "layers": 1,
                "kernel_size": 3,
                "transformer_heads": 2,
                "dropout": 0.0,
                "device": "cpu",
                "seed": 13,
            },
        )()

    def test_train_local_tcn_code_prior_writes_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            export_dir, manifest_path = self._write_codes_export(Path(tempdir))
            output_dir = export_dir / "local_tcn_prior"

            summary = train_code_prior(
                args=self._args("local_tcn"),
                export_dir=export_dir,
                manifest_path=manifest_path,
                eval_export_dir=None,
                eval_manifest_path=None,
                output_dir=output_dir,
            )

            self.assertEqual(summary["schema_version"], CODE_PRIOR_SCHEMA_VERSION)
            self.assertEqual(summary["prior_family"], "local_tcn")
            self.assertEqual(summary["context_scope"], "local")
            self.assertEqual(summary["token_ordering"], TOKEN_ORDERING)
            self.assertTrue((output_dir / "train_metrics.jsonl").exists())
            self.assertTrue((output_dir / "val_metrics.jsonl").exists())
            self.assertTrue((output_dir / "summary.json").exists())
            self.assertTrue((output_dir / "config.json").exists())
            self.assertTrue((output_dir / "checkpoint.pt").exists())
            self.assertEqual(len(summary["final_val"]["stage_bits_per_code"]), 2)
            self.assertIn("estimated_entropy_bitrate_kbps", summary["final_val"])

    def test_train_long_transformer_code_prior_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            export_dir, manifest_path = self._write_codes_export(Path(tempdir))
            output_dir = export_dir / "long_transformer_prior"

            summary = train_code_prior(
                args=self._args("long_transformer"),
                export_dir=export_dir,
                manifest_path=manifest_path,
                eval_export_dir=None,
                eval_manifest_path=None,
                output_dir=output_dir,
            )

            self.assertEqual(summary["prior_family"], "long_transformer")
            self.assertEqual(summary["context_scope"], "long")
            self.assertTrue((output_dir / "checkpoint.pt").exists())
            self.assertEqual(len(read_jsonl(output_dir / "val_metrics.jsonl")), 1)


class ContextPriorPipelineTest(unittest.TestCase):
    def test_pipeline_dry_run_prints_all_stages(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script = root / "scripts" / "run-context-prior-pipeline.sh"
        result = subprocess.run(
            [
                "bash",
                str(script),
                "--manifest",
                "evals/data/manifests/test.jsonl",
                "--checkpoint",
                "/tmp/checkpoint.pt",
                "--output-root",
                "/tmp/context-pipeline",
                "--export-dir",
                "/tmp/custom-export",
                "--max-items",
                "1",
                "--train-steps",
                "1",
                "--bundle-dir",
                "/tmp/context-bundle",
                "--dry-run",
            ],
            cwd=root,
            check=True,
            text=True,
            capture_output=True,
        )

        self.assertIn("export_neural_codec.py", result.stdout)
        self.assertIn("diagnose_representations.py", result.stdout)
        self.assertIn("evaluate_code_priors.py", result.stdout)
        self.assertIn("train_code_prior.py", result.stdout)
        self.assertIn("collect_context_results.py", result.stdout)
        self.assertIn("pack-context-results.sh", result.stdout)
        self.assertIn("local_tcn", result.stdout)
        self.assertIn("long_transformer", result.stdout)
        self.assertIn("--prior-root /tmp/context-pipeline/priors", result.stdout)
        self.assertIn("--bundle-dir /tmp/context-bundle", result.stdout)


class ContextResultsBundleTest(unittest.TestCase):
    def _write_context_outputs(self, output_root: Path) -> tuple[Path, Path, Path]:
        export_dir = output_root / "neural-4k-export"
        results_dir = output_root / "results"
        prior_root = output_root / "priors"
        analytic_dir = export_dir / "code_priors"
        diagnostics_dir = export_dir / "diagnostics"
        local_dir = prior_root / "local-tcn"
        long_dir = prior_root / "long-transformer"
        heavy_dirs = [
            export_dir / "representations",
            export_dir / "reconstructions",
            local_dir,
            long_dir,
            analytic_dir,
            diagnostics_dir,
            results_dir,
        ]
        for directory in heavy_dirs:
            directory.mkdir(parents=True, exist_ok=True)

        (export_dir / "manifest.jsonl").write_text('{"id": "utt"}\n')
        (export_dir / "run.json").write_text("{}\n")
        (diagnostics_dir / "summary.json").write_text("{}\n")
        (diagnostics_dir / "diagnostics.jsonl").write_text('{"stage": "diagnostics"}\n')
        for directory in [analytic_dir, local_dir, long_dir]:
            (directory / "summary.json").write_text("{}\n")
            (directory / "config.json").write_text("{}\n")
            (directory / "train_metrics.jsonl").write_text('{"split": "train"}\n')
            (directory / "val_metrics.jsonl").write_text('{"split": "val"}\n')
        (results_dir / "results.jsonl").write_text('{"stage": "result"}\n')
        (results_dir / "summary.csv").write_text("stage\nresult\n")
        (results_dir / "summary.json").write_text("{}\n")

        (export_dir / "representations" / "utt.codes.pt").write_text("heavy tensor\n")
        (export_dir / "reconstructions" / "utt.wav").write_text("heavy wav\n")
        (local_dir / "checkpoint.pt").write_text("heavy checkpoint\n")
        (long_dir / "checkpoint.pt").write_text("heavy checkpoint\n")
        return export_dir, results_dir, prior_root

    def test_pack_context_results_copies_only_lightweight_files(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script = root / "scripts" / "pack-context-results.sh"
        with tempfile.TemporaryDirectory() as tempdir:
            temp_root = Path(tempdir)
            output_root = temp_root / "context"
            export_dir, results_dir, prior_root = self._write_context_outputs(output_root)
            bundle_dir = temp_root / "bundle"

            result = subprocess.run(
                [
                    "bash",
                    str(script),
                    "--output-root",
                    str(output_root),
                    "--export-dir",
                    str(export_dir),
                    "--results-dir",
                    str(results_dir),
                    "--prior-root",
                    str(prior_root),
                    "--bundle-dir",
                    str(bundle_dir),
                ],
                cwd=root,
                check=True,
                text=True,
                capture_output=True,
            )

            self.assertIn("Wrote lightweight context results bundle", result.stdout)
            self.assertTrue((bundle_dir / "BUNDLE_MANIFEST.txt").exists())
            self.assertTrue((bundle_dir / "neural-4k-export" / "manifest.jsonl").exists())
            self.assertTrue((bundle_dir / "neural-4k-export" / "diagnostics" / "summary.json").exists())
            self.assertTrue((bundle_dir / "neural-4k-export" / "code_priors" / "summary.json").exists())
            self.assertTrue((bundle_dir / "results" / "summary.csv").exists())
            self.assertTrue((bundle_dir / "priors" / "local-tcn" / "summary.json").exists())
            self.assertTrue((bundle_dir / "priors" / "long-transformer" / "val_metrics.jsonl").exists())
            self.assertFalse((bundle_dir / "neural-4k-export" / "representations" / "utt.codes.pt").exists())
            self.assertFalse((bundle_dir / "neural-4k-export" / "reconstructions" / "utt.wav").exists())
            self.assertFalse((bundle_dir / "priors" / "local-tcn" / "checkpoint.pt").exists())

    def test_pack_context_results_dry_run_does_not_create_bundle(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script = root / "scripts" / "pack-context-results.sh"
        with tempfile.TemporaryDirectory() as tempdir:
            temp_root = Path(tempdir)
            output_root = temp_root / "context"
            export_dir, results_dir, prior_root = self._write_context_outputs(output_root)
            bundle_dir = temp_root / "dry-bundle"

            result = subprocess.run(
                [
                    "bash",
                    str(script),
                    "--output-root",
                    str(output_root),
                    "--export-dir",
                    str(export_dir),
                    "--results-dir",
                    str(results_dir),
                    "--prior-root",
                    str(prior_root),
                    "--bundle-dir",
                    str(bundle_dir),
                    "--dry-run",
                ],
                cwd=root,
                check=True,
                text=True,
                capture_output=True,
            )

            self.assertIn("[dry-run] copy", result.stdout)
            self.assertFalse(bundle_dir.exists())


class ContextResultsCollectionTest(unittest.TestCase):
    def test_collect_context_results_writes_table_and_gate_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            export_dir = root / "neural-4k-export"
            diagnostics_dir = export_dir / "diagnostics"
            analytic_dir = export_dir / "code_priors"
            local_dir = root / "priors" / "local-tcn"
            long_dir = root / "priors" / "long-transformer"
            diagnostics_dir.mkdir(parents=True)
            analytic_dir.mkdir(parents=True)
            local_dir.mkdir(parents=True)
            long_dir.mkdir(parents=True)

            (diagnostics_dir / "summary.json").write_text(
                """
{
  "schema_version": "representation-diagnostics-v1",
  "aggregates": [
    {"representation": "latent", "context_scope": "local", "items": 1, "metric": "predictability_score", "mean_value": 0.2},
    {"representation": "latent", "context_scope": "long", "items": 1, "metric": "predictability_score", "mean_value": 0.24},
    {"representation": "codes", "context_scope": "local", "items": 1, "metric": "window_reuse_rate", "mean_value": 0.3},
    {"representation": "codes", "context_scope": "full_utterance", "items": 1, "metric": "window_reuse_rate", "mean_value": 0.4}
  ],
  "gate_recommendations": [
    {"representation": "latent", "best_long_or_full_scope": "long", "passed": true},
    {"representation": "codes", "best_long_or_full_scope": "full_utterance", "passed": true}
  ]
}
""".strip()
            )
            (analytic_dir / "summary.json").write_text(
                """
{
  "schema_version": "code-prior-entropy-v1",
  "token_ordering": "time_major_frame_stage_coarse_to_fine",
  "evaluation_split": "self_eval",
  "priors": [
    {
      "prior_family": "unigram",
      "prior_name": "unigram_per_stage",
      "context_scope": "none",
      "context_window_frames": 0,
      "context_window_seconds": 0.0,
      "bits_per_code": 2.0,
      "stage_bits_per_code": [2.0, 2.0],
      "estimated_entropy_bitrate_kbps": 4.0,
      "nominal_bitrate_kbps": 4.0,
      "entropy_savings_ratio": 0.0,
      "relative_improvement_vs_unigram": 0.0
    },
    {
      "prior_family": "previous_frame",
      "prior_name": "previous_frame_markov_per_stage",
      "context_scope": "local",
      "context_window_frames": 1,
      "context_window_seconds": 0.5,
      "bits_per_code": 1.5,
      "stage_bits_per_code": [1.5, 1.5],
      "estimated_entropy_bitrate_kbps": 3.0,
      "nominal_bitrate_kbps": 4.0,
      "entropy_savings_ratio": 0.25,
      "relative_improvement_vs_unigram": 0.25
    }
  ]
}
""".strip()
            )
            (local_dir / "summary.json").write_text(
                """
{
  "schema_version": "code-prior-entropy-v1",
  "prior_family": "local_tcn",
  "prior_name": "local_tcn",
  "token_ordering": "time_major_frame_stage_coarse_to_fine",
  "context_scope": "local",
  "context_window_frames": 5,
  "context_window_seconds": 2.5,
  "nominal_bitrate_kbps": 4.0,
  "final_val": {
    "dataset_split": "self_eval",
    "bits_per_code": 1.4,
    "stage_bits_per_code": [1.4, 1.4],
    "estimated_entropy_bitrate_kbps": 2.8,
    "nominal_bitrate_kbps": 4.0,
    "entropy_savings_ratio": 0.3
  }
}
""".strip()
            )
            (long_dir / "summary.json").write_text(
                """
{
  "schema_version": "code-prior-entropy-v1",
  "prior_family": "long_transformer",
  "prior_name": "long_transformer",
  "token_ordering": "time_major_frame_stage_coarse_to_fine",
  "context_scope": "long",
  "context_window_frames": 32,
  "context_window_seconds": 16.0,
  "nominal_bitrate_kbps": 4.0,
  "final_val": {
    "dataset_split": "self_eval",
    "bits_per_code": 1.2,
    "stage_bits_per_code": [1.2, 1.2],
    "estimated_entropy_bitrate_kbps": 2.4,
    "nominal_bitrate_kbps": 4.0,
    "entropy_savings_ratio": 0.4
  }
}
""".strip()
            )

            rows, summary = collect_context_results(
                export_dir=export_dir,
                output_dir=root / "results",
                diagnostics_summary_path=diagnostics_dir / "summary.json",
                analytic_prior_summary_path=analytic_dir / "summary.json",
                trained_prior_summary_paths=[local_dir / "summary.json", long_dir / "summary.json"],
                gate_threshold=0.05,
            )

            self.assertEqual(summary["schema_version"], CONTEXT_RESULTS_SCHEMA_VERSION)
            self.assertTrue((root / "results" / "results.jsonl").exists())
            self.assertTrue((root / "results" / "summary.csv").exists())
            self.assertTrue((root / "results" / "summary.json").exists())
            self.assertEqual(len(rows), 8)
            self.assertTrue(summary["go_no_go"]["diagnostics_gate_passed"])
            self.assertTrue(summary["go_no_go"]["prior_gate_passed"])
            self.assertTrue(summary["go_no_go"]["go_to_codec_context_training"])
            long_rows = [row for row in rows if row["prior_family"] == "long_transformer"]
            self.assertEqual(len(long_rows), 1)
            self.assertEqual(long_rows[0]["gate_reference"], "best_local:local_tcn")
            self.assertTrue(long_rows[0]["gate_passed"])


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
