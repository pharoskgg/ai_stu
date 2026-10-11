"""优化器的参数更新、梯度清空和完整训练行为。"""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minitorch import Tensor, nn, optim


class OptimizerTests(unittest.TestCase):
    def test_base_step_is_not_implemented(self):
        optimizer = optim.Optimizer([Tensor([1.0])])
        with self.assertRaises(NotImplementedError):
            optimizer.step()

    def test_sgd_updates_data_in_place_and_preserves_gradient(self):
        parameter = nn.Parameter([1.0, 2.0])
        parameter.grad = np.array([2.0, -4.0])
        data, gradient = parameter.data, parameter.grad
        optimizer = optim.SGD([parameter], lr=0.1)
        optimizer.step()
        np.testing.assert_allclose(parameter.data, [0.8, 2.4])
        self.assertIs(optimizer.params[0], parameter)
        self.assertIs(parameter.data, data)
        self.assertIs(parameter.grad, gradient)
        np.testing.assert_array_equal(parameter.grad, [2.0, -4.0])
        self.assertIsNone(parameter.grad_fn)

    def test_default_learning_rate(self):
        parameter = Tensor([1.0])
        parameter.grad = np.array([2.0])
        optim.SGD([parameter]).step()
        np.testing.assert_allclose(parameter.data, [0.98])

    def test_generator_parameters_survive_multiple_steps(self):
        first, second = Tensor([1.0]), Tensor([2.0])
        first.grad, second.grad = np.array([2.0]), np.array([-4.0])
        optimizer = optim.SGD((p for p in (first, second)), lr=0.1)
        optimizer.step()
        optimizer.step()
        np.testing.assert_allclose(first.data, [0.6])
        np.testing.assert_allclose(second.data, [2.8])
        self.assertIs(optimizer.params[0], first)
        self.assertIs(optimizer.params[1], second)

    def test_shared_parameter_is_updated_once(self):
        parameter = Tensor([1.0])
        parameter.grad = np.array([2.0])
        optim.SGD([parameter, parameter], lr=0.1).step()
        np.testing.assert_allclose(parameter.data, [0.8])

    def test_equal_valued_distinct_parameters_are_both_updated(self):
        first, second = Tensor([1.0]), Tensor([1.0])
        first.grad, second.grad = np.array([2.0]), np.array([3.0])
        optim.SGD([first, second], lr=0.1).step()
        np.testing.assert_allclose(first.data, [0.8])
        np.testing.assert_allclose(second.data, [0.7])

    def test_missing_gradient_is_skipped(self):
        parameter = Tensor([1.0, 2.0])
        data = parameter.data
        optim.SGD([parameter], lr=0.1).step()
        np.testing.assert_array_equal(parameter.data, [1.0, 2.0])
        self.assertIs(parameter.data, data)
        self.assertIsNone(parameter.grad)

    def test_zero_gradient_leaves_data_unchanged(self):
        parameter = Tensor([1.0, 2.0])
        parameter.grad = np.zeros(2)
        optim.SGD([parameter], lr=0.1).step()
        np.testing.assert_array_equal(parameter.data, [1.0, 2.0])

    def test_existing_gradient_is_used_when_requires_grad_is_false(self):
        parameter = Tensor([1.0], requires_grad=False)
        parameter.grad = np.array([2.0])
        optim.SGD([parameter], lr=0.1).step()
        np.testing.assert_allclose(parameter.data, [0.8])

    def test_zero_grad_defaults_to_none(self):
        parameter = Tensor([1.0])
        parameter.grad = np.array([2.0])
        optim.Optimizer([parameter]).zero_grad()
        self.assertIsNone(parameter.grad)

    def test_zero_grad_false_zeros_existing_array_in_place(self):
        parameter, unused = Tensor([1.0, 2.0]), Tensor([3.0])
        parameter.grad = np.array([2.0, -4.0])
        gradient = parameter.grad
        optim.Optimizer([parameter, unused]).zero_grad(set_to_none=False)
        self.assertIs(parameter.grad, gradient)
        np.testing.assert_array_equal(gradient, [0.0, 0.0])
        self.assertIsNone(unused.grad)

    def test_empty_parameters_are_rejected(self):
        for constructor in (optim.Optimizer, optim.SGD):
            with self.subTest(constructor=constructor):
                with self.assertRaises(ValueError):
                    constructor(iter(()))

    def test_non_tensor_parameters_are_rejected(self):
        for constructor in (optim.Optimizer, optim.SGD):
            for value in (1.0, np.array([1.0]), None):
                with self.subTest(constructor=constructor, value=value):
                    with self.assertRaises(TypeError):
                        constructor([Tensor([1.0]), value])

    def test_non_leaf_parameters_are_rejected(self):
        output = Tensor([1.0], requires_grad=True) * 2
        for constructor in (optim.Optimizer, optim.SGD):
            with self.subTest(constructor=constructor):
                with self.assertRaises(ValueError):
                    constructor([output])

    def test_learning_rate_rejects_boolean_and_non_real_values(self):
        for learning_rate in (True, False, np.bool_(True), "0.1", None, 1j):
            with self.subTest(lr=learning_rate):
                with self.assertRaises(TypeError):
                    optim.SGD([Tensor([1.0])], lr=learning_rate)

    def test_learning_rate_rejects_negative_and_non_finite_values(self):
        for learning_rate in (-0.1, float("nan"), float("inf"), -float("inf")):
            with self.subTest(lr=learning_rate):
                with self.assertRaises(ValueError):
                    optim.SGD([Tensor([1.0])], lr=learning_rate)

    def test_zero_learning_rate_is_allowed(self):
        parameter = Tensor([1.0])
        parameter.grad = np.array([2.0])
        optim.SGD([parameter], lr=0).step()
        np.testing.assert_array_equal(parameter.data, [1.0])

    def test_numpy_real_learning_rate_is_allowed(self):
        parameter = Tensor([1.0])
        parameter.grad = np.array([2.0])
        optim.SGD([parameter], lr=np.float64(0.1)).step()
        np.testing.assert_allclose(parameter.data, [0.8])

    def test_linear_regression_converges(self):
        rng = np.random.default_rng(7)
        inputs = rng.normal(size=(64, 2))
        x = Tensor(inputs)
        target = Tensor(inputs @ np.array([[2.0], [-3.0]]) + 0.5)
        random_state = np.random.get_state()
        try:
            np.random.seed(7)
            model = nn.Linear(2, 1)
        finally:
            np.random.set_state(random_state)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        initial_loss = float(((model(x) - target) ** 2).mean().data)
        for _ in range(150):
            optimizer.zero_grad()
            loss = ((model(x) - target) ** 2).mean()
            loss.backward()
            optimizer.step()
        final_loss = float(((model(x) - target) ** 2).mean().data)
        self.assertLess(final_loss, initial_loss * 1e-6)
        self.assertLess(final_loss, 1e-8)
        np.testing.assert_allclose(model.weight.data, [[2.0], [-3.0]], atol=1e-4)
        np.testing.assert_allclose(model.bias.data, [0.5], atol=1e-4)


if __name__ == "__main__":
    unittest.main()
