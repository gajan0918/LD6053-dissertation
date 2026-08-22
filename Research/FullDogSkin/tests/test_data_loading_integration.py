import tempfile
import unittest
from pathlib import Path

from PIL import Image

import path_setup  # noqa: F401
from data_utils import get_data_loaders


def write_image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (80, 80), color=(40, 90, 130)).save(path)


class DataLoadingIntegrationTests(unittest.TestCase):
    def test_get_data_loaders_builds_all_splits_and_client_loaders(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir)
            for split in ("train", "valid", "test"):
                for class_name in ("Dermatitis", "Healthy"):
                    for index in range(2):
                        write_image(dataset_root / split / class_name / f"{index}.jpg")

            train_loader, client_loaders, valid_loader, test_loader, metadata = get_data_loaders(
                dataset_root,
                num_clients=2,
                batch_size=2,
                seed=7,
                num_workers=0,
                image_size=32,
                use_weighted_sampler=False,
            )

            self.assertEqual(metadata["num_classes"], 2)
            self.assertEqual(metadata["class_names"], ["Dermatitis", "Healthy"])
            self.assertEqual(len(train_loader.dataset), 4)
            self.assertEqual(len(valid_loader.dataset), 4)
            self.assertEqual(len(test_loader.dataset), 4)
            self.assertEqual(len(client_loaders), 2)
            self.assertEqual(sum(len(loader.dataset) for loader in client_loaders), 4)

            images, labels = next(iter(train_loader))
            self.assertEqual(tuple(images.shape), (2, 3, 32, 32))
            self.assertEqual(tuple(labels.shape), (2,))


if __name__ == "__main__":
    unittest.main()
