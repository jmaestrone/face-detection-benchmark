"""Tests for inference helper functions."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from face_detection_benchmark.cli import app
from face_detection_benchmark.inference import (
    iter_batches,
    load_frame_image_batch,
    load_image_batch,
    predict_egoblur_from_coco_dataset,
    predict_faces_from_coco_dataset,
    predict_faces_from_frames,
    rfdetr_model_name_from_weights,
)
from face_detection_benchmark.models.egoblur import (
    DEFAULT_EGOBLUR_CAMERA_NAME,
    DEFAULT_EGOBLUR_MODEL_NAME,
    DEFAULT_EGOBLUR_NMS_IOU_THRESHOLD,
    DEFAULT_EGOBLUR_RESIZE_MAX,
    DEFAULT_EGOBLUR_RESIZE_MIN,
    DEFAULT_EGOBLUR_THRESHOLD,
    EgoBlurConfig,
    EgoBlurDetector,
    egoblur_face_detections_to_records,
    resolve_egoblur_device,
)
from face_detection_benchmark.models.insightface import (
    DEFAULT_INSIGHTFACE_CTX_ID,
    DEFAULT_INSIGHTFACE_DET_SIZE,
    DEFAULT_INSIGHTFACE_MODEL_PACK,
    DEFAULT_INSIGHTFACE_PROVIDERS,
    DEFAULT_INSIGHTFACE_THRESHOLD,
    InsightFaceConfig,
    InsightFaceDetector,
    detector_output_to_records,
    parse_providers,
)
from face_detection_benchmark.models.rfdetr import RfdetrTrainingConfig, train_rfdetr
from face_detection_benchmark.predictions import (
    DetectionRecord,
    ImagePredictionRecord,
    prediction_record_to_json,
    summarize_latency,
    write_latency_summary,
)
from typer.testing import CliRunner


class InferenceHelpersTest(unittest.TestCase):
    """Coverage for lightweight inference helpers."""

    def test_rfdetr_model_name_from_weights(self) -> None:
        """Build a readable model name from an RF-DETR checkpoint path."""
        self.assertEqual(
            rfdetr_model_name_from_weights(Path("models/checkpoint_best_ema_2.pth")),
            "rfdetr-checkpoint-best-ema-2",
        )

    def test_public_image_loader_alias_remains_compatible(self) -> None:
        """Keep the legacy public image batch loader alias available."""
        self.assertIs(load_image_batch, load_frame_image_batch)

    def test_iter_batches_yields_fixed_size_groups(self) -> None:
        """Split records into fixed-size batches without dropping leftovers."""
        self.assertEqual(
            list(iter_batches([1, 2, 3, 4, 5], batch_size=2)),
            [[1, 2], [3, 4], [5]],
        )

    def test_frame_prediction_validates_inputs_before_loading_model(self) -> None:
        """Reject missing frame inputs before constructing the RF-DETR adapter."""
        with self.assertRaisesRegex(ValueError, "Frames directory does not exist"):
            predict_faces_from_frames(
                frames_dir=Path("missing-frames"),
                weights_path=Path("missing-weights.pth"),
            )

    def test_benchmark_prediction_validates_inputs_before_loading_model(self) -> None:
        """Reject missing benchmark inputs before constructing the RF-DETR adapter."""
        with self.assertRaisesRegex(ValueError, "Dataset directory does not exist"):
            predict_faces_from_coco_dataset(
                dataset_dir=Path("missing-dataset"),
                weights_path=Path("missing-weights.pth"),
            )

    def test_parse_insightface_providers(self) -> None:
        """Parse comma-separated ONNX Runtime provider names."""
        self.assertEqual(
            parse_providers("CUDAExecutionProvider, CPUExecutionProvider"),
            ("CUDAExecutionProvider", "CPUExecutionProvider"),
        )
        self.assertEqual(
            parse_providers(["CUDAExecutionProvider", " "]),
            ("CUDAExecutionProvider",),
        )

    def test_insightface_detection_output_to_records(self) -> None:
        """Normalize InsightFace detector output boxes into prediction records."""
        records = detector_output_to_records(
            (
                [
                    [1, 2, 11, 22, 0.9876543],
                    [3, 4, 5, 6, 0.1],
                ],
                None,
            )
        )

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].bbox_xyxy, [1.0, 2.0, 11.0, 22.0])
        self.assertEqual(records[0].bbox_xywh, [1.0, 2.0, 10.0, 20.0])
        self.assertEqual(records[0].confidence, 0.987654)
        self.assertIsNone(records[0].class_id)
        self.assertEqual(detector_output_to_records(None), [])

    def test_insightface_missing_dependency_message(self) -> None:
        """Explain how to install the optional InsightFace dependency."""
        real_import = __import__

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name.startswith("insightface"):
                raise ImportError("missing insightface")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaisesRegex(ValueError, "uv sync --extra insightface"):
                InsightFaceDetector(
                    InsightFaceConfig(
                        model_pack=DEFAULT_INSIGHTFACE_MODEL_PACK,
                        providers=DEFAULT_INSIGHTFACE_PROVIDERS,
                        ctx_id=DEFAULT_INSIGHTFACE_CTX_ID,
                        det_size=DEFAULT_INSIGHTFACE_DET_SIZE,
                        threshold=DEFAULT_INSIGHTFACE_THRESHOLD,
                    )
                )

    def test_egoblur_face_detections_to_records(self) -> None:
        """Normalize EgoBlur face boxes and scores into prediction records."""

        @dataclass
        class EgoBlurOutput:
            face_bboxes: list[list[float]]
            face_scores: list[float]
            lp_bboxes: list[list[float]]

        records = egoblur_face_detections_to_records(
            EgoBlurOutput(
                face_bboxes=[[1, 2, 11, 22], [3, 4, 5, 6]],
                face_scores=[0.9876543, 0.1],
                lp_bboxes=[[50, 50, 60, 60]],
            )
        )

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].bbox_xyxy, [1.0, 2.0, 11.0, 22.0])
        self.assertEqual(records[0].bbox_xywh, [1.0, 2.0, 10.0, 20.0])
        self.assertEqual(records[0].confidence, 0.987654)
        self.assertEqual(records[0].class_id, 0)
        self.assertEqual(records[0].class_name, "Human face")
        self.assertEqual(egoblur_face_detections_to_records(None), [])

    def test_egoblur_missing_dependency_message(self) -> None:
        """Explain how to install the optional EgoBlur dependency."""
        real_import = __import__

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name.startswith("gen2"):
                raise ImportError("missing egoblur")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaisesRegex(ValueError, "uv sync --extra egoblur"):
                EgoBlurDetector(
                    EgoBlurConfig(
                        face_model_path=Path("models/egoblur/ego_blur_face_gen2.jit"),
                        camera_name=DEFAULT_EGOBLUR_CAMERA_NAME,
                        threshold=DEFAULT_EGOBLUR_THRESHOLD,
                        nms_iou_threshold=DEFAULT_EGOBLUR_NMS_IOU_THRESHOLD,
                        device="cpu",
                        resize_min=DEFAULT_EGOBLUR_RESIZE_MIN,
                        resize_max=DEFAULT_EGOBLUR_RESIZE_MAX,
                    )
                )

    def test_resolve_egoblur_device_rejects_mps(self) -> None:
        """Keep EgoBlur v1 device support limited to CUDA and CPU."""
        with self.assertRaisesRegex(ValueError, "auto, cuda, cpu"):
            resolve_egoblur_device("mps")

    def test_egoblur_benchmark_prediction_validates_inputs_before_loading_model(
        self,
    ) -> None:
        """Reject missing EgoBlur benchmark inputs before loading the model."""
        with self.assertRaisesRegex(ValueError, "Dataset directory does not exist"):
            predict_egoblur_from_coco_dataset(
                dataset_dir=Path("missing-dataset"),
                face_model_path=Path("missing-model.jit"),
            )

    def test_egoblur_benchmark_cli_rejects_missing_weights(self) -> None:
        """Surface missing EgoBlur weights through the CLI."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "dataset"
            dataset_dir.mkdir()
            result = CliRunner().invoke(
                app,
                [
                    "predict-egoblur-benchmark",
                    "--dataset-dir",
                    str(dataset_dir),
                    "--face-model-path",
                    str(Path(temp_dir) / "missing.jit"),
                    "--device",
                    "cpu",
                ],
            )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("EgoBlur face model file does not exist", result.output)

    def test_egoblur_benchmark_cli_rejects_invalid_resize(self) -> None:
        """Reject invalid EgoBlur resize values at the CLI boundary."""
        result = CliRunner().invoke(
            app,
            [
                "predict-egoblur-benchmark",
                "--resize-min",
                "0",
            ],
        )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Invalid value for '--resize-min'", result.output)

    def test_prediction_json_includes_timing_and_model_metadata(self) -> None:
        """Serialize normalized prediction rows with latency and backend metadata."""
        record = ImagePredictionRecord(
            file_name="image.jpg",
            image_path="dataset/image.jpg",
            width=100,
            height=100,
            detections=[
                DetectionRecord(
                    bbox_xyxy=[1.0, 2.0, 11.0, 22.0],
                    bbox_xywh=[1.0, 2.0, 10.0, 20.0],
                    confidence=0.9,
                    class_id=1,
                )
            ],
            model_name="rfdetr-test",
            model_config={
                "checkpoint_name": "checkpoint.pth",
                "device": "cpu",
                "threshold": 0.005,
                "max_detections": 40,
            },
            threshold=0.005,
            device="cpu",
            backend="rfdetr",
            timing_ms={"inference": 12.3456},
        )

        payload = prediction_record_to_json(record)

        self.assertEqual(payload["timing_ms"], {"inference": 12.3456})
        self.assertEqual(payload["model_config"]["checkpoint_name"], "checkpoint.pth")
        self.assertEqual(payload["model_config"]["device"], "cpu")
        self.assertEqual(payload["detections"][0]["confidence"], 0.9)

    def test_latency_summary_reports_distribution(self) -> None:
        """Summarize benchmark inference latency without accuracy metrics."""
        summary = summarize_latency(
            model_name="insightface-test",
            backend="insightface",
            device="cpu",
            image_count=4,
            detection_count=7,
            inference_times_ms=[10.0, 20.0, 30.0, 40.0],
            total_runtime_ms=125.4321,
            model_config={
                "providers": ["CPUExecutionProvider"],
                "ctx_id": -1,
                "det_size": 960,
                "threshold": 0.005,
                "model_pack": "buffalo_l",
            },
        )

        self.assertEqual(summary["model_name"], "insightface-test")
        self.assertEqual(summary["backend"], "insightface")
        self.assertEqual(summary["image_count"], 4)
        self.assertEqual(summary["detection_count"], 7)
        self.assertEqual(summary["total_runtime_ms"], 125.4321)
        self.assertEqual(summary["total_inference_ms"], 100.0)
        self.assertEqual(summary["per_image_inference_ms"]["mean"], 25.0)
        self.assertEqual(summary["per_image_inference_ms"]["median"], 25.0)
        self.assertEqual(summary["per_image_inference_ms"]["p90"], 37.0)
        self.assertNotIn("precision", summary)
        self.assertNotIn("map_50_95", summary)

    def test_write_latency_summary_writes_json_and_csv(self) -> None:
        """Write latency artifacts with stable JSON and CSV fields."""
        summary = summarize_latency(
            model_name="rfdetr-test",
            backend="rfdetr",
            device="cpu",
            image_count=1,
            detection_count=2,
            inference_times_ms=[15.0],
            total_runtime_ms=20.0,
            model_config={"checkpoint_name": "checkpoint.pth"},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            latency_path = root / "latency" / "rfdetr-test.json"
            latency_table_path = root / "latency.csv"

            write_latency_summary(
                summary=summary,
                latency_path=latency_path,
                latency_table_path=latency_table_path,
            )

            payload = json.loads(latency_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["model_name"], "rfdetr-test")
            self.assertEqual(payload["per_image_inference_ms"]["max"], 15.0)
            csv_lines = latency_table_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(csv_lines), 2)
            self.assertIn("model_name,backend,device", csv_lines[0])
            self.assertIn("rfdetr-test,rfdetr,cpu", csv_lines[1])

    def test_egoblur_prediction_json_includes_metadata(self) -> None:
        """Serialize EgoBlur prediction rows with face-only metadata."""
        record = ImagePredictionRecord(
            file_name="image.jpg",
            image_path="dataset/image.jpg",
            width=100,
            height=100,
            detections=[
                DetectionRecord(
                    bbox_xyxy=[1.0, 2.0, 11.0, 22.0],
                    bbox_xywh=[1.0, 2.0, 10.0, 20.0],
                    confidence=0.9,
                    class_id=0,
                    class_name="Human face",
                )
            ],
            model_name=DEFAULT_EGOBLUR_MODEL_NAME,
            model_config={
                "generation": "gen2",
                "face_model_name": "ego_blur_face_gen2.jit",
                "camera_name": "camera-rgb",
                "threshold": 0.005,
                "nms_iou_threshold": 0.5,
                "device": "cpu",
                "resize_min": 1200,
                "resize_max": 1200,
            },
            threshold=0.005,
            device="cpu",
            backend="egoblur",
            timing_ms={"inference": 12.3456},
        )

        payload = prediction_record_to_json(record)

        self.assertEqual(payload["model_name"], DEFAULT_EGOBLUR_MODEL_NAME)
        self.assertEqual(payload["backend"], "egoblur")
        self.assertEqual(payload["model_config"]["generation"], "gen2")
        self.assertEqual(payload["model_config"]["camera_name"], "camera-rgb")
        self.assertEqual(payload["model_config"]["resize_min"], 1200)
        self.assertEqual(payload["model_config"]["resize_max"], 1200)
        self.assertEqual(payload["detections"][0]["class_name"], "Human face")
        self.assertEqual(payload["detections"][0]["class_id"], 0)

    def test_train_rfdetr_rejects_benchmark_dataset_dir(self) -> None:
        """Prevent benchmark datasets from being used as training inputs."""
        with self.assertRaisesRegex(ValueError, "must not point inside data/benchmark"):
            train_rfdetr(
                RfdetrTrainingConfig(
                    dataset_dir=Path("data/benchmark/target-video-test-3fps-clean"),
                    output_dir=Path("runs/training/test"),
                    epochs=1,
                    batch_size=1,
                    device="cpu",
                )
            )

    def test_train_rfdetr_writes_reproducibility_artifacts(self) -> None:
        """Write config and metadata before invoking the RF-DETR training API."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "training-dataset"
            output_dir = root / "training-output"
            weights_path = root / "checkpoint.pth"
            dataset_dir.mkdir()
            weights_path.write_bytes(b"weights")

            with patch("rfdetr.RFDETRLarge") as model_class:
                result = train_rfdetr(
                    RfdetrTrainingConfig(
                        dataset_dir=dataset_dir,
                        output_dir=output_dir,
                        weights_path=weights_path,
                        epochs=2,
                        batch_size=3,
                        device="cpu",
                        dataset_file="roboflow",
                        num_workers=0,
                    )
                )

            self.assertEqual(result.output_dir, output_dir)
            model_class.assert_called_once_with(pretrain_weights=str(weights_path))
            model_class.return_value.train.assert_called_once()
            train_kwargs = model_class.return_value.train.call_args.kwargs
            self.assertEqual(train_kwargs["dataset_dir"], str(dataset_dir))
            self.assertEqual(train_kwargs["output_dir"], str(output_dir))
            self.assertEqual(train_kwargs["epochs"], 2)
            self.assertEqual(train_kwargs["batch_size"], 3)
            self.assertEqual(train_kwargs["device"], "cpu")
            self.assertEqual(train_kwargs["dataset_file"], "roboflow")
            self.assertEqual(train_kwargs["num_workers"], 0)
            self.assertEqual(
                train_kwargs["notes"]["weights_path"], weights_path.as_posix()
            )

            config_payload = json.loads(result.config_path.read_text(encoding="utf-8"))
            metadata_payload = json.loads(
                result.metadata_path.read_text(encoding="utf-8")
            )
            self.assertEqual(config_payload["dataset_dir"], dataset_dir.as_posix())
            self.assertEqual(config_payload["output_dir"], output_dir.as_posix())
            self.assertEqual(config_payload["epochs"], 2)
            self.assertEqual(metadata_payload["status"], "completed")
            self.assertIn("rfdetr", metadata_payload["package_versions"])

    def test_train_rfdetr_resolves_auto_device_before_training_api(self) -> None:
        """Convert the CLI auto device value before calling RF-DETR training."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "training-dataset"
            output_dir = root / "training-output"
            dataset_dir.mkdir()

            with (
                patch("torch.cuda.is_available", return_value=False),
                patch("torch.backends.mps.is_available", return_value=False),
                patch("rfdetr.RFDETRLarge") as model_class,
            ):
                result = train_rfdetr(
                    RfdetrTrainingConfig(
                        dataset_dir=dataset_dir,
                        output_dir=output_dir,
                        epochs=1,
                        batch_size=1,
                        device="auto",
                        num_workers=0,
                    )
                )

            train_kwargs = model_class.return_value.train.call_args.kwargs
            self.assertEqual(train_kwargs["device"], "cpu")
            self.assertEqual(train_kwargs["notes"]["requested_device"], "auto")
            self.assertEqual(train_kwargs["notes"]["resolved_device"], "cpu")
            config_payload = json.loads(result.config_path.read_text(encoding="utf-8"))
            self.assertEqual(config_payload["device"], "auto")
            self.assertEqual(config_payload["resolved_device"], "cpu")

    def test_train_rfdetr_cli_requires_explicit_dataset_dir(self) -> None:
        """Require callers to supply a training dataset path."""
        result = CliRunner().invoke(
            app,
            [
                "train-rfdetr",
                "--output-dir",
                "runs/training/test",
                "--epochs",
                "1",
            ],
        )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Missing option '--dataset-dir'", result.output)

    def test_train_rfdetr_cli_rejects_benchmark_dataset_dir(self) -> None:
        """Surface benchmark dataset guardrails through the CLI."""
        result = CliRunner().invoke(
            app,
            [
                "train-rfdetr",
                "--dataset-dir",
                "data/benchmark/target-video-test-3fps-clean",
                "--output-dir",
                "runs/training/test",
                "--epochs",
                "1",
                "--batch-size",
                "1",
                "--device",
                "cpu",
            ],
        )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("must not point inside data/benchmark", result.output)


if __name__ == "__main__":
    unittest.main()
