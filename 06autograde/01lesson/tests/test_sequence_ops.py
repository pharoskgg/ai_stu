"""序列选取、堆叠及共享输入梯度。"""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import minitorch
from minitorch import Tensor


class SequenceOpsTests(unittest.TestCase):
    def test_select_negative_axis_and_index_scatter_gradient(self):
        x = Tensor([[1., 2., 3.], [4., 5., 6.]], requires_grad=True)
        self.assertTrue(hasattr(x, "select"), "缺少 Tensor.select")
        y = x.select(dim=-1, index=-2)
        np.testing.assert_array_equal(y.data, [2., 5.])
        y.backward([2., 3.])
        np.testing.assert_array_equal(x.grad, [[0., 2., 0.], [0., 3., 0.]])

    def test_select_scalar_result_and_invalid_indices(self):
        x = Tensor([1., 2.], requires_grad=True)
        self.assertTrue(hasattr(x, "select"), "缺少 Tensor.select")
        x.select(0, 1).backward()
        np.testing.assert_array_equal(x.grad, [0., 1.])
        for dim, index in ((1, 0), (0, 2), (0, -3)):
            with self.subTest(dim=dim, index=index), self.assertRaises(IndexError):
                x.select(dim, index)
        for dim, index in ((True, 0), (0, True), (0, 0.5)):
            with self.subTest(dim=dim, index=index), self.assertRaises(TypeError):
                x.select(dim, index)
        with self.assertRaises(IndexError):
            Tensor(1.).select(0, 0)
        with self.assertRaises(IndexError):
            Tensor(np.empty((0, 2))).select(0, 0)

    def test_stack_shared_inputs_and_nonzero_axis_gradient(self):
        self.assertTrue(hasattr(minitorch, "stack"), "缺少 minitorch.stack")
        x = Tensor([1., 2.], requires_grad=True)
        y = minitorch.stack([x, Tensor([3., 4.]), x], dim=-1)
        np.testing.assert_array_equal(y.data, [[1., 3., 1.], [2., 4., 2.]])
        y.backward([[1., 2., 3.], [4., 5., 6.]])
        np.testing.assert_array_equal(x.grad, [4., 10.])

    def test_stack_scalar_and_constant_inputs(self):
        self.assertTrue(hasattr(minitorch, "stack"), "缺少 minitorch.stack")
        y = minitorch.stack((Tensor(1.), Tensor(2.)))
        np.testing.assert_array_equal(y.data, [1., 2.])
        self.assertFalse(y.requires_grad)
        self.assertIsNone(y.grad_fn)

    def test_stack_rejects_empty_mismatched_and_invalid_inputs(self):
        self.assertTrue(hasattr(minitorch, "stack"), "缺少 minitorch.stack")
        for values in ([], [Tensor([1.]), Tensor([1., 2.])]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                minitorch.stack(values)
        for values in (Tensor([1.]), [Tensor([1.]), [2.]]):
            with self.subTest(values=values), self.assertRaises(TypeError):
                minitorch.stack(values)
        for dim in (-3, 2):
            with self.subTest(dim=dim), self.assertRaises(IndexError):
                minitorch.stack([Tensor([1.])], dim=dim)
        with self.assertRaises(TypeError):
            minitorch.stack([Tensor([1.])], dim=True)


if __name__ == "__main__":
    unittest.main()
