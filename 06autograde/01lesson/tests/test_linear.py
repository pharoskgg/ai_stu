"""Linear 的前向、参数管理和梯度验证。"""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minitorch import Tensor, nn


class LinearTests(unittest.TestCase):
    def test_forward_with_bias(self):
        layer = nn.Linear(3, 2)
        layer.weight.data[:] = [[1., 2.], [3., 4.], [5., 6.]]
        layer.bias.data[:] = [0.5, -0.5]
        x = Tensor([[1., 0., 2.], [-1., 3., 1.]])
        np.testing.assert_allclose(layer(x).data, [[11.5, 13.5], [13.5, 15.5]])

    def test_initialization_and_parameter_registration(self):
        layer = nn.Linear(3, 2)
        self.assertIsInstance(layer, nn.Module)
        self.assertEqual(layer.weight.data.shape, (3, 2))
        self.assertEqual(layer.bias.data.shape, (2,))
        self.assertEqual([name for name, _ in layer.named_parameters()], ["weight", "bias"])
        bound = 1 / np.sqrt(3)
        for parameter in layer.parameters():
            self.assertIsInstance(parameter, nn.Parameter)
            self.assertTrue(parameter.requires_grad)
            self.assertIsNone(parameter.grad_fn)
            self.assertTrue(np.isfinite(parameter.data).all())
            self.assertTrue((np.abs(parameter.data) <= bound).all())
        self.assertTrue(np.any(layer.weight.data != 0))

    def test_forward_without_bias_and_backward(self):
        layer = nn.Linear(2, 1, bias=False)
        layer.weight.data[:] = [[1.], [2.]]
        self.assertIsNone(layer.bias)
        self.assertEqual([name for name, _ in layer.named_parameters()], ["weight"])
        y = layer(Tensor([[1., 2.], [3., 4.]]))
        np.testing.assert_allclose(y.data, [[5.], [11.]])
        y.sum().backward()
        np.testing.assert_allclose(layer.weight.grad, [[4.], [6.]])

    def test_input_weight_and_bias_gradients(self):
        layer = nn.Linear(2, 1)
        layer.weight.data[:] = [[1.], [2.]]
        layer.bias.data[:] = [0.5]
        x = Tensor([[1., 2.], [3., 4.]], requires_grad=True)
        (layer(x) ** 2).mean().backward()
        np.testing.assert_allclose(x.grad, [[5.5, 11.], [11.5, 23.]])
        np.testing.assert_allclose(layer.weight.grad, [[40.], [57.]])
        np.testing.assert_allclose(layer.bias.grad, [17.])

    def test_higher_dimensional_forward(self):
        layer = nn.Linear(2, 1)
        layer.weight.data[:] = [[1.], [2.]]
        layer.bias.data[:] = [0.5]
        x = Tensor([[[[1., 2.], [3., 4.]]], [[[5., 6.], [7., 8.]]]])
        np.testing.assert_allclose(layer(x).data, [[[[5.5], [11.5]]], [[[17.5], [23.5]]]])

    def test_batched_gradients_match_finite_differences(self):
        layer = nn.Linear(2, 2)
        layer.weight.data[:] = [[0.2, -0.3], [0.4, 0.5]]
        layer.bias.data[:] = [0.1, -0.2]
        x = Tensor([[[1., 2.], [-1., 0.5]], [[3., -2.], [0.2, 0.7]]], requires_grad=True)
        (layer(x) ** 2).mean().backward()
        # 中心差分只读取前向数值，独立检查输入梯度和跨批次参数梯度。
        epsilon = 1e-6
        for tensor in (x, layer.weight, layer.bias):
            numeric = np.empty_like(tensor.data)
            for index in np.ndindex(tensor.data.shape):
                original = tensor.data[index]
                tensor.data[index] = original + epsilon
                plus = (layer(x).data ** 2).mean()
                tensor.data[index] = original - epsilon
                minus = (layer(x).data ** 2).mean()
                tensor.data[index] = original
                numeric[index] = (plus - minus) / (2 * epsilon)
            np.testing.assert_allclose(tensor.grad, numeric, rtol=1e-6, atol=1e-8)

    def test_invalid_feature_dimensions(self):
        for dimensions in ((0, 2), (2, 0), (-1, 2), (2, -1)):
            with self.subTest(dimensions=dimensions):
                with self.assertRaises(ValueError):
                    nn.Linear(*dimensions)
        for dimensions in ((1.5, 2), (2, "3"), (True, 2), (2, False)):
            with self.subTest(dimensions=dimensions):
                with self.assertRaises(TypeError):
                    nn.Linear(*dimensions)

    def test_invalid_input_shapes(self):
        layer = nn.Linear(3, 2)
        for data in (1., [1., 2., 3.], [[1., 2.]], np.zeros((2, 1, 4))):
            with self.subTest(shape=np.shape(data)):
                with self.assertRaises(ValueError):
                    layer(Tensor(data))


if __name__ == "__main__":
    unittest.main()
