import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import path_setup  # noqa: F401
from federated_utils import average_params, get_params, set_params
from train_utils import evaluate_model


class FederatedAndTrainingUnitTests(unittest.TestCase):
    def test_average_params_calculates_weighted_average(self):
        client_params = [
            {"weight": torch.tensor([1.0, 3.0])},
            {"weight": torch.tensor([5.0, 7.0])},
        ]

        averaged = average_params(client_params, weights=[0.25, 0.75])

        self.assertTrue(torch.allclose(averaged["weight"], torch.tensor([4.0, 6.0])))

    def test_get_and_set_params_round_trip_model_state(self):
        source = nn.Linear(2, 2)
        target = nn.Linear(2, 2)

        with torch.no_grad():
            source.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
            source.bias.copy_(torch.tensor([0.5, -0.5]))

        set_params(target, get_params(source), torch.device("cpu"))

        self.assertTrue(torch.allclose(target.weight, source.weight))
        self.assertTrue(torch.allclose(target.bias, source.bias))

    def test_evaluate_model_reports_accuracy_and_confusion_matrix(self):
        model = nn.Linear(2, 2)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
            model.bias.zero_()

        inputs = torch.tensor([
            [2.0, 0.0],
            [0.0, 2.0],
            [0.0, 2.0],
            [2.0, 0.0],
        ])
        labels = torch.tensor([0, 1, 0, 0])
        loader = DataLoader(TensorDataset(inputs, labels), batch_size=2)

        metrics = evaluate_model(
            model,
            loader,
            torch.device("cpu"),
            num_classes=2,
            class_names=["class_a", "class_b"],
        )

        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["confusion_matrix"], [[2, 1], [0, 1]])
        self.assertEqual(metrics["per_class_accuracy"]["class_a"], 2 / 3)
        self.assertEqual(metrics["per_class_accuracy"]["class_b"], 1.0)


if __name__ == "__main__":
    unittest.main()
