"""Tensor 形状查询、变换及反向传播的行为测试。"""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minitorch import Tensor


class ShapeTests(unittest.TestCase):
    def test_shape_queries_for_scalar_empty_and_batched_tensors(self):
        for data, shape, count in ((3., (), 1), (np.empty((2, 0, 3)), (2, 0, 3), 0),
                                   (np.ones((2, 3, 4)), (2, 3, 4), 24)):
            with self.subTest(shape=shape):
                x = Tensor(data)
                self.assertEqual(x.shape, shape)
                self.assertEqual(x.size(), shape)
                self.assertEqual(x.ndim, len(shape))
                self.assertEqual(x.numel(), count)
        x = Tensor(np.ones((2, 3, 4)))
        self.assertEqual(x.size(dim=-1), 4)
        self.assertEqual(x.size(np.int64(1)), 3)
        self.assertEqual(x.size(-3), 2)

    def test_size_rejects_invalid_dimensions(self):
        x = Tensor(np.ones((2, 3)))
        for dim in (True, np.bool_(False), 1., '0', (0,)):
            with self.subTest(dim=dim), self.assertRaises(TypeError):
                x.size(dim)
        for dim in (-3, 2):
            with self.subTest(dim=dim), self.assertRaises(IndexError):
                x.size(dim)
        for dim in (-1, 0):
            with self.subTest(scalar_dim=dim), self.assertRaises(IndexError):
                Tensor(1.).size(dim)

    def test_reshape_accepts_variadic_tuple_list_and_inferred_shape(self):
        for shape in ((3, 2), ((3, 2),), ([3, 2],), (-1, 2), (3, -1),
                      (np.int64(3), np.int64(2))):
            with self.subTest(shape=shape):
                x = Tensor([[1., 2., 3.], [4., 5., 6.]], requires_grad=True)
                y = x.reshape(*shape)
                np.testing.assert_array_equal(y.data, [[1., 2.], [3., 4.], [5., 6.]])
                y.backward([[1., 3.], [5., 7.], [9., 11.]])
                np.testing.assert_array_equal(x.grad, [[1., 3., 5.], [7., 9., 11.]])

    def test_reshape_scalar_and_empty_tensor(self):
        x = Tensor(4., requires_grad=True)
        y = x.reshape(1, 1).reshape(())
        self.assertEqual(y.shape, ())
        y.backward()
        np.testing.assert_array_equal(x.grad, np.array(1.))
        for shape in ((0, 3), (2, -1), (-1,)):
            with self.subTest(shape=shape):
                x = Tensor(np.empty((0, 2)), requires_grad=True)
                y = x.reshape(*shape)
                self.assertEqual(y.shape, (0, 3) if shape == (0, 3) else
                                 (2, 0) if shape == (2, -1) else (0,))
                y.sum().backward()
                self.assertEqual(x.grad.shape, (0, 2))

    def test_reshape_rejects_invalid_shape_types(self):
        for shape in ((), (True,), (2., 3), ('6',), (None,), ([2, False],),
                      (np.bool_(True),), ((2, 3), 1)):
            with self.subTest(shape=shape), self.assertRaises(TypeError):
                Tensor(np.arange(6.)).reshape(*shape)

    def test_reshape_rejects_incompatible_or_ambiguous_shapes(self):
        for shape in ((4, 2), (-1, -1), (-2, 3), (0, 6), ((),)):
            with self.subTest(shape=shape), self.assertRaises(RuntimeError):
                Tensor(np.arange(6.)).reshape(*shape)
        with self.assertRaises(RuntimeError):
            Tensor([]).reshape(0, -1)

    def test_transpose_swaps_two_dimensions_and_their_gradients(self):
        for dims in ((0, 1), (-2, -1), (np.int64(0), np.int64(1))):
            with self.subTest(dims=dims):
                x = Tensor([[1., 2., 3.], [4., 5., 6.]], requires_grad=True)
                y = x.transpose(dim0=dims[0], dim1=dims[1])
                np.testing.assert_array_equal(y.data, [[1., 4.], [2., 5.], [3., 6.]])
                y.backward([[1., 2.], [3., 4.], [5., 6.]])
                np.testing.assert_array_equal(x.grad, [[1., 3., 5.], [2., 4., 6.]])

    def test_transpose_preserves_other_batch_dimensions(self):
        x = Tensor([[[1., 2., 3.]], [[4., 5., 6.]]], requires_grad=True)
        y = x.transpose(0, -1)
        np.testing.assert_array_equal(y.data, [[[1., 4.]], [[2., 5.]], [[3., 6.]]])
        y.backward([[[2., 3.]], [[4., 5.]], [[6., 7.]]])
        np.testing.assert_array_equal(x.grad, [[[2., 4., 6.]], [[3., 5., 7.]]])

    def test_transpose_scalar_vector_equal_axes_and_empty_tensor(self):
        for data, dims in ((5., (0, -1)), ([2., 3.], (0, -1)),
                           ([[1., 2.]], (1, 1)), (np.empty((2, 0)), (0, 1))):
            with self.subTest(dims=dims, shape=np.shape(data)):
                x = Tensor(data, requires_grad=True)
                y = x.transpose(*dims)
                self.assertEqual(y.shape, (0, 2) if x.numel() == 0 else x.shape)
                y.sum().backward()
                np.testing.assert_array_equal(x.grad, np.ones_like(data))

    def test_transpose_rejects_invalid_dimensions(self):
        for dims in ((True, 0), (0, np.bool_(False)), (1., 0), (0, None)):
            with self.subTest(dims=dims), self.assertRaises(TypeError):
                Tensor(np.ones((2, 3))).transpose(*dims)
        for data, dims in ((np.ones((2, 3)), (2, 0)), (np.ones((2, 3)), (0, -3)),
                           (1., (1, 0)), (1., (0, -2))):
            with self.subTest(dims=dims), self.assertRaises(IndexError):
                Tensor(data).transpose(*dims)

    def test_shape_operations_on_constants_do_not_build_graph(self):
        x = Tensor([[1., 2.], [3., 4.]])
        for y in (x.reshape(-1), x.transpose(0, 1)):
            self.assertFalse(y.requires_grad)
            self.assertIsNone(y.grad_fn)

    def test_shape_results_preserve_existing_independent_storage_behavior(self):
        x = Tensor([[1., 2.], [3., 4.]])
        for y in (x.reshape(-1), x.transpose(0, 1)):
            y.data.fill(0)
            np.testing.assert_array_equal(x.data, [[1., 2.], [3., 4.]])

    def test_shared_paths_and_repeated_backward_accumulate(self):
        x = Tensor([[1., 2.], [3., 4.]], requires_grad=True)
        y = x.transpose(0, 1).reshape(-1)
        loss = (y * Tensor([1., 2., 3., 4.])).sum() + y.sum()
        loss.backward()
        np.testing.assert_array_equal(x.grad, [[2., 4.], [3., 5.]])
        loss.backward()
        np.testing.assert_array_equal(x.grad, [[4., 8.], [6., 10.]])

    def test_shape_chain_with_matmul_matches_finite_differences(self):
        data = np.arange(1., 13.).reshape(2, 2, 3) / 10
        weight = np.array([[1., -2.], [0.5, 3.], [-1., 2.], [2., 0.5]])
        x = Tensor(data, requires_grad=True)
        loss = ((x.transpose(0, 2).reshape(3, 4) @ Tensor(weight)) ** 2).mean()
        loss.backward()
        numerical = np.zeros_like(data)
        epsilon = 1e-6
        # 用独立的 NumPy 前向和中心差分检查整条链的梯度。
        def objective(value):
            return np.mean((np.swapaxes(value, 0, 2).reshape(3, 4) @ weight) ** 2)
        for index in np.ndindex(data.shape):
            plus, minus = data.copy(), data.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            numerical[index] = (objective(plus) - objective(minus)) / (2 * epsilon)
        np.testing.assert_allclose(x.grad, numerical, rtol=1e-6, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
