"""Sequential 的有序执行、模块管理和多层网络训练。"""

from collections import OrderedDict
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minitorch import Tensor, nn, optim
from minitorch.nn import modules


class SequentialTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(nn, "Sequential"), "nn.Sequential 尚未实现")

    def test_module_namespace_exports_sequential(self):
        self.assertIs(modules.Sequential, nn.Sequential)

    def test_positional_modules_have_numeric_names_and_execute_in_order(self):
        first, second = nn.Linear(1, 1), nn.Linear(1, 1)
        first.weight.data[:] = 2
        first.bias.data[:] = 1
        second.weight.data[:] = 3
        second.bias.data[:] = -1
        model = nn.Sequential(first, second)
        self.assertEqual(list(model.named_children()), [("0", first), ("1", second)])
        np.testing.assert_allclose(model(Tensor([[2.]])).data, [[14.]])

    def test_ordered_dict_preserves_names_and_forward_order(self):
        linear, relu = nn.Linear(1, 1), nn.ReLU()
        linear.weight.data[:] = -2
        linear.bias.data[:] = 1
        model = nn.Sequential(OrderedDict([("projection", linear), ("activation", relu)]))
        self.assertEqual(list(model.named_children()), [("projection", linear), ("activation", relu)])
        self.assertIs(model.projection, linear)
        np.testing.assert_allclose(model(Tensor([[-1.], [1.]])).data, [[3.], [0.]])

    def test_empty_container_returns_the_same_input(self):
        value = Tensor([[1., -2.]])
        model = nn.Sequential()
        self.assertIs(model(value), value)
        self.assertEqual(len(model), 0)
        self.assertEqual(list(model), [])
        self.assertEqual(list(model.parameters()), [])

    def test_invalid_positional_modules_are_rejected(self):
        for value in (None, 1, "relu", Tensor([1.]), [nn.ReLU()], {"relu": nn.ReLU()}):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    nn.Sequential(nn.ReLU(), value)

    def test_invalid_named_modules_are_rejected(self):
        for value in (None, 1, Tensor([1.])):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    nn.Sequential(OrderedDict([("invalid", value)]))

    def test_named_modules_reuse_registration_name_validation(self):
        for name in ("", "a.b", "forward"):
            with self.subTest(name=name):
                with self.assertRaises(KeyError):
                    nn.Sequential(OrderedDict([(name, nn.ReLU())]))
        with self.assertRaises(TypeError):
            nn.Sequential(OrderedDict([(1, nn.ReLU())]))

    def test_length_iteration_and_positive_negative_indices(self):
        first, last = nn.ReLU(), nn.Linear(1, 1)
        model = nn.Sequential(first, last, first)
        self.assertEqual(len(model), 3)
        self.assertEqual(list(model), [first, last, first])
        for index, expected in ((0, first), (1, last), (2, first), (-1, first), (-2, last), (-3, first)):
            with self.subTest(index=index):
                self.assertIs(model[index], expected)
        self.assertIs(model[np.int64(1)], last)

    def test_out_of_range_indices_are_rejected(self):
        for model, indices in ((nn.Sequential(), (0, -1)), (nn.Sequential(nn.ReLU()), (1, -2))):
            for index in indices:
                with self.subTest(index=index, length=len(model)):
                    with self.assertRaises(IndexError):
                        model[index]

    def test_non_integer_indices_are_rejected(self):
        model = nn.Sequential(nn.ReLU())
        for index in ("0", 0.0, None, slice(None), True, False, np.bool_(True)):
            with self.subTest(index=index):
                with self.assertRaises(TypeError):
                    model[index]

    def test_nested_parameters_have_recursive_registered_names(self):
        first, last = nn.Linear(2, 3), nn.Linear(3, 1)
        model = nn.Sequential(OrderedDict([
            ("hidden", nn.Sequential(first, nn.ReLU())), ("output", last)
        ]))
        named = list(model.named_parameters())
        self.assertEqual([name for name, _ in named], ["hidden.0.weight", "hidden.0.bias", "output.weight", "output.bias"])
        self.assertEqual([param for _, param in named], [first.weight, first.bias, last.weight, last.bias])

    def test_shared_module_executes_each_position_but_parameters_are_unique(self):
        layer = nn.Linear(1, 1)
        layer.weight.data[:] = 2
        layer.bias.data[:] = 1
        model = nn.Sequential(layer, layer)
        output = model(Tensor([[1.]]))
        np.testing.assert_allclose(output.data, [[7.]])
        output.sum().backward()
        np.testing.assert_allclose(layer.weight.grad, [[5.]])
        np.testing.assert_allclose(layer.bias.grad, [3.])
        self.assertEqual(list(model.parameters()), [layer.weight, layer.bias])
        self.assertEqual([name for name, _ in model.named_parameters()], ["0.weight", "0.bias"])

    def test_train_eval_and_zero_grad_reach_nested_layers(self):
        inner = nn.Sequential(nn.Linear(1, 1), nn.ReLU())
        model = nn.Sequential(inner, nn.Linear(1, 1))
        self.assertIs(model.eval(), model)
        self.assertTrue(all(not module.training for module in model.modules()))
        self.assertIs(model.train(), model)
        self.assertTrue(all(module.training for module in model.modules()))
        model(Tensor([[1.], [2.]])).sum().backward()
        gradients = [param.grad for param in model.parameters()]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        model.zero_grad(set_to_none=False)
        for param, gradient in zip(model.parameters(), gradients):
            self.assertIs(param.grad, gradient)
            np.testing.assert_array_equal(gradient, np.zeros_like(gradient))
        model.zero_grad()
        self.assertTrue(all(param.grad is None for param in model.parameters()))

    def test_linear_relu_linear_training_reduces_loss_with_finite_gradients(self):
        inputs = np.random.default_rng(7).normal(size=(64, 2))
        x = Tensor(inputs)
        target = Tensor((1.5 * np.maximum(inputs[:, 0], 0)
                         - 0.7 * np.maximum(-inputs[:, 1], 0) + 0.2)[:, None])
        random_state = np.random.get_state()
        try:
            np.random.seed(7)
            model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))
        finally:
            np.random.set_state(random_state)
        optimizer = optim.SGD(model.parameters(), lr=0.05)
        initial_loss = float(((model(x) - target) ** 2).mean().data)
        for _ in range(300):
            optimizer.zero_grad()
            loss = ((model(x) - target) ** 2).mean()
            loss.backward()
            for name, param in model.named_parameters():
                with self.subTest(parameter=name):
                    self.assertIsNotNone(param.grad)
                    self.assertTrue(np.isfinite(param.grad).all())
            optimizer.step()
        final_loss = float(((model(x) - target) ** 2).mean().data)
        self.assertLess(final_loss, initial_loss * 0.05)


if __name__ == "__main__":
    unittest.main()
