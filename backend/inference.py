"""Inference pipeline for foul detection.

Runs the multi-task model over a single video clip and returns one decoded
prediction per task.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

import config
from models.ActionData import ActionData
from models.Decoder import Decoder
from utils.FeatureExtractor import FeatureExtractor
from utils.FoulDataPreprocessor import FoulDataPreprocessor
from utils.HDF5Reader import save_to_hdf5
from utils.MultiTaskModel import (
    ImprovedMultiTaskModel,
    build_default_model,
    load_model,
)

logger = logging.getLogger(__name__)

# The clip is always attributed to this camera, which is the one the model was
# trained on and the only camera FoulDataPreprocessor keeps.
TARGET_CAMERA = 'Close-up player or field referee'


def confidence_bar(probability: float, width: int = 20) -> str:
    """ASCII confidence bar.

    Deliberately avoids block-drawing characters such as U+2591/U+2593: the
    original version raised UnicodeEncodeError on Windows consoles, which use
    cp1252 by default.
    """
    filled = max(0, min(width, int(probability * width)))
    return "#" * filled + "-" * (width - filled)


class InferencePipeline:
    """Pipeline for running inference with the improved foul detection model."""

    def __init__(self, model_path: str = None, base_dir: str = None):
        self.base_dir = Path(base_dir) if base_dir else config.BASE_DIR
        self.preprocessor = FoulDataPreprocessor()
        self.decoder = Decoder()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.metadata, self.history = self._load_model(
            model_path or str(config.MODEL_PATH)
        )
        self.weights_loaded = self.metadata.get("weights_loaded", False)
        self.extractor = FeatureExtractor(
            base_dir=str(self.base_dir),
            model_type=self.metadata.get(
                "feature_config", {}
            ).get("model_type", config.FEATURE_EXTRACTOR_MODEL),
        )
        self._apply_finetuned_backbone()

        logger.info("Pipeline ready. Device: %s", self.device)
        self._log_architecture()

    def _apply_finetuned_backbone(self) -> None:
        """Use fine-tuned backbone weights when the checkpoint carries them.

        A fine-tuned checkpoint stores the backbone alongside the heads. Without
        this, inference would silently use the generic Kinetics weights and the
        served predictions would not match the trained model.
        """
        state = self.metadata.get("backbone_state_dict")
        if not state:
            if self.metadata.get("feature_config", {}).get("finetuned"):
                logger.warning(
                    "Checkpoint is marked finetuned but carries no backbone weights; "
                    "falling back to the generic %s backbone.",
                    config.FEATURE_EXTRACTOR_MODEL,
                )
            return
        self.extractor.load_finetuned_backbone(state)
        logger.info("Serving with a FINE-TUNED %s backbone",
                    self.metadata.get("feature_config", {}).get("model_type", "r3d_18"))

    def _log_architecture(self) -> None:
        logger.info("Model architecture:")
        for task, head in self.model.primary_heads.items():
            logger.info("- %s: %d classes", task, head.net[-1].out_features)
        logger.info("- severity: %d classes", self.model.severity_head[-1].out_features)

    def _load_model(self, model_path: str) -> Tuple[ImprovedMultiTaskModel, dict, dict]:
        """Load the trained model, falling back to an untrained model.

        The pretrained checkpoint is gitignored and is not distributed with the
        repository, so a fresh clone cannot load real weights. Rather than
        crashing on import we build the same architecture with random weights
        and flag it, so the API stays up and a real checkpoint can be dropped in
        later via the FOUL_MODEL_PATH environment variable.
        """
        model_path = str(model_path)
        if os.path.exists(model_path):
            try:
                model, metadata, history = load_model(model_path, self.device)
                metadata['weights_loaded'] = True
                logger.info("Loaded trained checkpoint from %s", model_path)
                return model, metadata, history
            except Exception as exc:
                logger.error("Failed to load checkpoint %s: %s", model_path, exc)
                if not config.ALLOW_UNTRAINED_FALLBACK:
                    raise

        if not config.ALLOW_UNTRAINED_FALLBACK:
            raise FileNotFoundError(
                f"Model file not found: {model_path}. Train a checkpoint or set "
                "FOUL_MODEL_PATH, or set ALLOW_UNTRAINED_FALLBACK=true to serve "
                "with randomly initialised weights."
            )

        logger.warning("=" * 72)
        logger.warning("NO TRAINED CHECKPOINT FOUND at: %s", model_path)
        logger.warning("Serving an UNTRAINED model with randomly initialised weights.")
        logger.warning("Predictions from this model are meaningless.")
        logger.warning("To fix: train with backend/training/training_pipeline.py, or")
        logger.warning("point FOUL_MODEL_PATH at an existing foul_detection_model.pth.")
        logger.warning("=" * 72)

        model, config_dict = build_default_model(self.device, seed=config.FALLBACK_SEED)
        return model, {
            'model_config': config_dict,
            'architecture_version': '2.0',
            'weights_loaded': False,
            'untrained_fallback': True,
        }, None

    def process_video_for_inference(self, video_path: str, replay_speed: float = 1.0) -> str:
        """Process a single video for inference and return the features file path."""
        output_dir = config.INFERENCE_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / 'inference_features.h5'

        logger.info("Processing video: %s", video_path)

        # Build a placeholder action so the existing HDF5 + preprocessing path
        # (which is shared with training) can be reused unchanged.
        dummy_action = ActionData({
            'Offence': 'Offence',
            'Bodypart': 'Under body',
            'Action class': 'Tackling',
            'Touch ball': 'No',
            'Try to play': 'Yes',
            'Severity': '1.0',
            'Upper body part': '',
            'Clips': []
        })

        features = self.extractor.extract_video_features(video_path)

        dummy_action.clips = [{
            'video_features': features,
            'Replay speed': replay_speed,
            'Camera type': TARGET_CAMERA
        }]

        logger.info("Features extracted successfully")

        save_to_hdf5([dummy_action], str(output_file))
        logger.info("Saved features to: %s", output_file)

        return str(output_file)

    def inference(self, inference_file: str) -> List[Dict]:
        """Run inference on processed video data."""
        logger.info("Starting inference...")

        if not os.path.exists(inference_file):
            raise FileNotFoundError(f"Inference file not found: {inference_file}")

        processed_data = self.preprocessor.process_data(inference_file)
        if processed_data is None:
            raise ValueError("No usable features produced from the video")

        X_test, _ = processed_data
        if X_test is None:
            raise ValueError("X_test is None after preprocessing")

        expected = self.metadata['model_config']['input_size']
        if X_test.shape[1] != expected:
            raise ValueError(
                f"Feature size mismatch: the model expects {expected} inputs but "
                f"the extractor produced {X_test.shape[1]}. Check that "
                "FEATURE_MODEL_TYPE matches the one used during training."
            )

        X_test = X_test.to(self.device).float()
        logger.info("Input shape for inference: %s", tuple(X_test.shape))

        with torch.no_grad():
            outputs = self.model(X_test)
            predictions = {
                task: F.softmax(output, dim=1).cpu().numpy()
                for task, output in outputs.items()
            }

        task_to_map = {
            'actionclass': self.decoder.action_class_map,
            'bodypart': self.decoder.bodypart_map,
            'offence': self.decoder.offence_map,
            'touchball': self.decoder.touchball_map,
            'trytoplay': self.decoder.trytoplay_map,
            'severity': self.decoder.severity_map,
        }

        display_names = {
            'actionclass': 'Action Class',
            'bodypart': 'Body Part',
            'offence': 'Offence',
            'touchball': 'Touch Ball',
            'trytoplay': 'Try to Play',
            'severity': 'Severity',
        }

        decoded_predictions = []
        for task, probs in predictions.items():
            task_map = task_to_map.get(task)
            if task_map is None:
                logger.warning("No decoder map for task %s; skipping", task)
                continue

            top_prob, top_idx = torch.max(torch.from_numpy(probs[0]), dim=0)
            class_idx = int(top_idx.item())
            if class_idx not in task_map:
                # Happens if a checkpoint has more head outputs than the map.
                logger.warning(
                    "Class index %d for task %s is not in the decoder map; skipping",
                    class_idx, task
                )
                continue

            decoded_predictions.append({
                'category': display_names.get(task, task),
                'prediction': task_map[class_idx].decode('utf-8'),
                'probability': float(top_prob.item()),
            })

        if not decoded_predictions:
            raise ValueError("Model produced no decodable predictions")

        return decoded_predictions


def main():
    """Run the inference pipeline on a test video."""
    video_path = os.getenv(
        "TEST_VIDEO",
        str(config.DATA_ROOT / "dataset" / "inference" / "testaction5_clip1.mp4"),
    )
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Test video not found: {video_path}")

    pipeline = InferencePipeline()

    print("\nDecoded Predictions with Probabilities:")
    print("=" * 50)
    features_path = pipeline.process_video_for_inference(video_path, replay_speed=1.4)
    for category in pipeline.inference(features_path):
        pct = category['probability']
        bar = confidence_bar(pct)
        print(f"\n{category['category'].upper()}:")
        print(f"  - {category['prediction']:<26} [{bar}] {pct:.1%}")

    logger.info("Inference pipeline completed successfully!")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    main()
