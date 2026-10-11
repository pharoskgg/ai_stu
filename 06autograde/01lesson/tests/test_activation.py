"""ReLU 的算子、函数、模块接口及组合网络梯度。"""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minitorch import Tensor, nn, ops


class ReLUTests(unittest.TestCase):
    def relu(self, tensor):
        self.assertTrue(hasattr(nn, "functional"), "缺少 nn.functional 接口")
        self.assertTrue(hasattr(nn.functional, "relu"), "缺少 functional.relu 接口")
        return nn.functional.relu(tensor)

    def test_operator_forward_and_nonuniform_upstream_gradient(self):
        self.assertTrue(hasattr(ops, "ReLU"), "缺少 ops.ReLU 算子")
        x = Tensor([-2., 0., 3.], requires_grad=True)
        y = ops.ReLU.apply(x)
        np.testing.assert_array_equal(y.data, [0., 0., 3.])
        y.backward([2., 3., 4.])
        np.testing.assert_array_equal(x.grad, [0., 0., 4.])

    def test_scalar_forward_and_backward(self):
        for value, output, gradient in ((-2., 0., 0.), (0., 0., 0.), (3., 3., 2.)):
            with self.subTest(value=value):
                x = Tensor(value, requires_grad=True)
                y = self.relu(x)
                self.assertEqual(y.data.shape, ())
                self.assertEqual(y.data.item(), output)
                y.backward(2.)
                self.assertEqual(x.grad.shape, ())
                self.assertEqual(x.grad.item(), gradient)

    def test_batched_shape_and_backward(self):
        x = Tensor([[[-2., 0., 3.]], [[4., -1., 2.]]], requires_grad=True)
        y = self.relu(x)
        self.assertEqual(y.data.shape, (2, 1, 3))
        np.testing.assert_array_equal(y.data, [[[0., 0., 3.]], [[4., 0., 2.]]])
        y.sum().backward()
        np.testing.assert_array_equal(x.grad, [[[0., 0., 1.]], [[1., 0., 1.]]])

    def test_input_is_unchanged_and_output_has_independent_data(self):
        x = Tensor([-2., 0., 3.])
        y = self.relu(x)
        np.testing.assert_array_equal(x.data, [-2., 0., 3.])
        y.data[:] = 10.
        np.testing.assert_array_equal(x.data, [-2., 0., 3.])

    def test_constant_does_not_build_graph(self):
        y = self.relu(Tensor([-2., 0., 3.]))
        np.testing.assert_array_equal(y.data, [0., 0., 3.])
        self.assertFalse(y.requires_grad)
        self.assertIsNone(y.grad_fn)

    def test_backward_uses_forward_mask_after_input_mutation(self):
        x = Tensor([-2., 0., 3.], requires_grad=True)
        y = self.relu(x)
        x.data[:] = [2., 3., -4.]
        y.backward([2., 3., 4.])
        np.testing.assert_array_equal(x.grad, [0., 0., 4.])

    def test_shared_relu_output_accumulates_branch_gradients(self):
        x = Tensor([-2., 0., 3.], requires_grad=True)
        y = self.relu(x)
        (y * 2. + y * 3.).sum().backward()
        np.testing.assert_array_equal(x.grad, [0., 0., 5.])

    def test_relu_paths_accumulate_on_shared_input(self):
        x = Tensor([-2., 0., 3.], requires_grad=True)
        (self.relu(x) * 2. + self.relu(x) * 3.).sum().backward()
        np.testing.assert_array_equal(x.grad, [0., 0., 5.])

    def test_module_has_no_parameters_and_matches_functional(self):
        self.assertTrue(hasattr(nn, "ReLU"), "缺少 nn.ReLU 模块")
        module = nn.ReLU()
        self.assertIsInstance(module, nn.Module)
        self.assertEqual(list(module.parameters()), [])
        x = Tensor([-2., 0., 3.], requires_grad=True)
        y = module(x)
        np.testing.assert_array_equal(y.data, [0., 0., 3.])
        np.testing.assert_array_equal(y.data, self.relu(x).data)
        y.backward([2., 3., 4.])
        np.testing.assert_array_equal(x.grad, [0., 0., 4.])

    def test_linear_relu_linear_backward(self):
        self.assertTrue(hasattr(nn, "ReLU"), "缺少 nn.ReLU 模块")
        first, activation, second = nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1)
        first.weight.data[:] = [[1., -1.], [1., 1.]]
        first.bias.data[:] = [0., 1.]
        second.weight.data[:] = [[2.], [-1.]]
        second.bias.data[:] = [0.5]
        x = Tensor([[1., -2.], [-1., 2.]], requires_grad=True)
        y = second(activation(first(x)))
        np.testing.assert_array_equal(y.data, [[0.5], [-1.5]])
        y.sum().backward()
        np.testing.assert_array_equal(x.grad, [[0., 0.], [3., 1.]])
        np.testing.assert_array_equal(first.weight.grad, [[-2., 1.], [4., -2.]])
        np.testing.assert_array_equal(first.bias.grad, [2., -1.])
        np.testing.assert_array_equal(second.weight.grad, [[1.], [4.]])
        np.testing.assert_array_equal(second.bias.grad, [2.])


if __name__ == "__main__":
    unittest.main()
