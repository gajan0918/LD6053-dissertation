import io
import json
import os
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from werkzeug.datastructures import FileStorage

import path_setup  # noqa: F401

os.environ.setdefault("ENABLE_CONTENT_VALIDATOR", "0")

import api


class FixedLogitModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("logits", torch.tensor(logits, dtype=torch.float32))

    def forward(self, batch):
        return self.logits.to(batch.device).unsqueeze(0).repeat(batch.shape[0], 1)


def build_image_storage(filename="dog.jpg", size=(96, 96), fmt="JPEG"):
    buffer = io.BytesIO()
    Image.new("RGB", size, color=(120, 90, 70)).save(buffer, format=fmt)
    buffer.seek(0)
    return FileStorage(stream=buffer, filename=filename, content_type=f"image/{fmt.lower()}")


class ApiIntegrationTests(unittest.TestCase):
    def setUp(self):
        names = [
            "model",
            "class_names",
            "disease_data",
            "none_class_idx",
            "startup_error",
            "content_validator_model",
            "content_validator_transform",
            "content_validator_categories",
            "LOW_CONFIDENCE_THRESHOLD",
            "SOFTMAX_TEMPERATURE",
            "PREDICTION_TTA_MODE",
            "ENABLE_NONE_GUARD",
            "NONE_REJECTION_THRESHOLD",
            "ENABLE_EXPLAINABILITY",
        ]
        self._original_state = {name: getattr(api, name) for name in names}
        self.addCleanup(self._restore_api_state)

        api.class_names = ["Dermatitis", "None", "Healthy"]
        api.none_class_idx = 1
        api.model = FixedLogitModel([8.0, -2.0, 0.5])
        api.startup_error = None
        api.disease_data = {
            "dog_skin_diseases": [
                {
                    "name": "Dermatitis",
                    "description": "Inflamed skin.",
                    "symptoms": ["itching"],
                    "causes": ["allergy"],
                    "treatment": ["vet care"],
                    "when_to_see_vet": "Book a vet visit if symptoms persist.",
                }
            ]
        }
        api.content_validator_model = None
        api.content_validator_transform = None
        api.content_validator_categories = None
        api.LOW_CONFIDENCE_THRESHOLD = 0.75
        api.SOFTMAX_TEMPERATURE = 1.0
        api.PREDICTION_TTA_MODE = "none"
        api.ENABLE_NONE_GUARD = False
        api.ENABLE_EXPLAINABILITY = False

    def _restore_api_state(self):
        for name, value in self._original_state.items():
            setattr(api, name, value)

    def test_health_endpoint_reports_ready_service(self):
        with api.app.test_client() as client:
            response = client.get("/health")

        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["status"], "healthy")
        self.assertTrue(payload["ready"])
        self.assertEqual(payload["num_classes"], 3)

    def test_predict_endpoint_returns_success_for_valid_image(self):
        image = build_image_storage()

        with api.app.test_client() as client:
            response = client.post(
                "/predict",
                data={"image": (image.stream, image.filename)},
                content_type="multipart/form-data",
            )

        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["prediction"], "Dermatitis")
        self.assertEqual(payload["description"], "Inflamed skin.")
        self.assertGreater(payload["confidence"], 99.0)
        self.assertEqual(payload["top_predictions"][0]["label"], "Dermatitis")

    def test_predict_endpoint_rejects_missing_image(self):
        with api.app.test_client() as client:
            response = client.post("/predict", data={})

        payload = response.get_json()
        self.assertEqual(response.status_code, 400)
        self.assertEqual(payload["status"], "bad_request")

    def test_predict_endpoint_rejects_invalid_content_when_none_probability_is_high(self):
        api.model = FixedLogitModel([0.1, 5.0, 0.2])
        api.ENABLE_NONE_GUARD = True
        api.NONE_REJECTION_THRESHOLD = 0.5
        image = build_image_storage()

        with api.app.test_client() as client:
            response = client.post(
                "/predict",
                data={"image": (image.stream, image.filename)},
                content_type="multipart/form-data",
            )

        payload = response.get_json()
        self.assertEqual(response.status_code, 422)
        self.assertEqual(payload["status"], "invalid_content")
        self.assertEqual(payload["prediction"], "None")

    def test_load_class_names_file_supports_report_payloads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "classes.json"
            path.write_text(
                json.dumps({"confusion_matrix": {"labels": ["A", "B"]}}),
                encoding="utf-8",
            )

            self.assertEqual(api.load_class_names_file(path), ["A", "B"])

    def test_load_request_image_validates_extension_and_dimensions(self):
        with self.assertRaisesRegex(ValueError, "Unsupported file type"):
            api.load_request_image(build_image_storage(filename="dog.gif"))

        with self.assertRaisesRegex(ValueError, "Image dimensions are invalid"):
            api.load_request_image(build_image_storage(size=(16, 16)))

        image = api.load_request_image(build_image_storage(size=(96, 96)))
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (96, 96))

    def test_adapt_checkpoint_output_layer_trims_one_extra_classifier_row(self):
        class ModelStateStub:
            def state_dict(self):
                return {
                    "classifier.1.weight": torch.zeros(3, 4),
                    "classifier.1.bias": torch.zeros(3),
                }

        checkpoint = {
            "classifier.1.weight": torch.arange(16, dtype=torch.float32).reshape(4, 4),
            "classifier.1.bias": torch.arange(4, dtype=torch.float32),
            "features.0.weight": torch.ones(1),
        }

        adapted = api.adapt_checkpoint_output_layer(checkpoint, ModelStateStub())

        self.assertEqual(tuple(adapted["classifier.1.weight"].shape), (3, 4))
        self.assertTrue(torch.equal(adapted["classifier.1.weight"], checkpoint["classifier.1.weight"][:3]))
        self.assertTrue(torch.equal(adapted["classifier.1.bias"], checkpoint["classifier.1.bias"][:3]))
        self.assertTrue(torch.equal(adapted["features.0.weight"], checkpoint["features.0.weight"]))


if __name__ == "__main__":
    unittest.main()
