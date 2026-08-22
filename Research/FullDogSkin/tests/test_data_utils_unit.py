import unittest

import torch

import path_setup  # noqa: F401
from data_utils import compute_class_weights, create_subset_sampler, split_indices_stratified


class DataUtilsUnitTests(unittest.TestCase):
    def test_split_indices_stratified_preserves_all_indices_once(self):
        targets = [0, 0, 0, 0, 1, 1, 1, 2]

        client_indices = split_indices_stratified(targets, num_clients=3, seed=123)
        flattened = [index for indices in client_indices for index in indices]

        self.assertEqual(sorted(flattened), list(range(len(targets))))
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(len(client_indices), 3)
        self.assertTrue(all(indices for indices in client_indices))

    def test_split_indices_stratified_is_reproducible_for_same_seed(self):
        targets = [0, 0, 1, 1, 2, 2, 2, 2]

        first = split_indices_stratified(targets, num_clients=2, seed=99)
        second = split_indices_stratified(targets, num_clients=2, seed=99)

        self.assertEqual(first, second)

    def test_compute_class_weights_gives_more_weight_to_rare_classes(self):
        weights = compute_class_weights(
            targets=[0, 0, 0, 0, 1, 1, 2],
            num_classes=3,
            power=1.0,
        )

        self.assertAlmostEqual(sum(weights) / len(weights), 1.0)
        self.assertGreater(weights[2], weights[1])
        self.assertGreater(weights[1], weights[0])

    def test_create_subset_sampler_uses_class_weight_per_sample(self):
        class DatasetStub:
            targets = [0, 1, 1, 2]

        sampler = create_subset_sampler(
            DatasetStub(),
            indices=[0, 2, 3],
            class_weights=[0.5, 1.0, 2.0],
        )

        self.assertEqual(sampler.num_samples, 3)
        self.assertTrue(torch.equal(sampler.weights, torch.DoubleTensor([0.5, 1.0, 2.0])))


if __name__ == "__main__":
    unittest.main()
