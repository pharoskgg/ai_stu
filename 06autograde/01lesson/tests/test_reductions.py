"""归约算子的前向值、梯度及迁移后的接口检查。"""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minitorch import Tensor


class ReductionBehaviorTests(unittest.TestCase):
    def test_global_reductions(self):
        for name, expected_value, expected_grad in (
            ("sum", 10.0, np.ones((2, 2))),
            ("mean", 2.5, np.full((2, 2), 0.25)),
        ):
            with self.subTest(name=name):
                x = Tensor([[1., 2.], [3., 4.]], requires_grad=True)
                y = getattr(x, name)()
                self.assertEqual(y.data.shape, ())
                np.testing.assert_allclose(y.data, expected_value)
                y.backward()
                np.testing.assert_allclose(x.grad, expected_grad)

    def test_dimension_reductions_with_weighted_seed(self):
        data = np.arange(24.).reshape(2, 3, 4)
        seed0 = np.arange(1., 13.).reshape(3, 4)
        seed1 = np.arange(1., 9.).reshape(2, 4)
        seed2 = np.arange(1., 7.).reshape(2, 3)
        cases = (
            (0, seed0, np.stack([seed0, seed0]), 2),
            (1, seed1, np.stack([seed1] * 3, axis=1), 3),
            (np.int64(1), seed1, np.stack([seed1] * 3, axis=1), 3),
            (-1, seed2, np.repeat(seed2[..., None], 4, axis=2), 4),
            ((0, 2), np.array([1., 2., 3.]),
             np.tile(np.array([1., 2., 3.])[None, :, None], (2, 1, 4)), 8),
            ((-1, 0), np.array([1., 2., 3.]),
             np.tile(np.array([1., 2., 3.])[None, :, None], (2, 1, 4)), 8),
            ((), data + 1, data + 1, 1),
        )
        for name in ("sum", "mean"):
            for keepdim in (False, True):
                for dim, seed, expected_grad, count in cases:
                    with self.subTest(name=name, dim=dim, keepdim=keepdim):
                        x = Tensor(data, requires_grad=True)
                        y = getattr(x, name)(dim=dim, keepdim=keepdim)
                        expected_value = getattr(np, name)(data, axis=dim, keepdims=keepdim)
                        self.assertEqual(y.data.shape, expected_value.shape)
                        np.testing.assert_allclose(y.data, expected_value)
                        y.backward(seed.reshape(y.data.shape))
                        np.testing.assert_allclose(x.grad, expected_grad / (count if name == "mean" else 1))

    def test_global_keepdim(self):
        for name, expected_gradient in (("sum", 3.0), ("mean", 0.5)):
            with self.subTest(name=name):
                x = Tensor(np.arange(6.).reshape(2, 3), requires_grad=True)
                y = getattr(x, name)(keepdim=True)
                self.assertEqual(y.data.shape, (1, 1))
                y.backward([[3.]])
                np.testing.assert_allclose(x.grad, np.full((2, 3), expected_gradient))

    def test_scalar_reductions(self):
        for name in ("sum", "mean"):
            for dim in ((None, 0, -1, ()) if name == "sum" else (None, ())):
                for keepdim in (False, True):
                    with self.subTest(name=name, dim=dim, keepdim=keepdim):
                        x = Tensor(3.0, requires_grad=True)
                        y = getattr(x, name)(dim=dim, keepdim=keepdim)
                        np.testing.assert_allclose(y.data, 3.0)
                        y.backward()
                        self.assertEqual(x.grad.shape, ())
                        np.testing.assert_allclose(x.grad, 1.0)

    def test_scalar_mean_explicit_axis_preserves_existing_error(self):
        # 原实现直接调用 np.mean，标量的显式轴会被 NumPy 拒绝。
        for dim in (0, -1):
            for keepdim in (False, True):
                with self.subTest(dim=dim, keepdim=keepdim):
                    with self.assertRaises(IndexError):
                        Tensor(3.0, requires_grad=True).mean(dim=dim, keepdim=keepdim)

    def test_mse_loss_and_repeated_backward(self):
        prediction = Tensor([1., 3.], requires_grad=True)
        target = Tensor([0., 1.])
        loss = ((prediction - target) ** 2).mean()
        np.testing.assert_allclose(loss.data, 2.5)
        loss.backward()
        np.testing.assert_allclose(prediction.grad, [1., 2.])
        self.assertIsNone(target.grad)
        loss.backward()
        np.testing.assert_allclose(prediction.grad, [2., 4.])

    def test_shared_reduction_paths(self):
        x = Tensor([1., 2., 3.], requires_grad=True)
        (x.sum() + x.mean()).backward()
        np.testing.assert_allclose(x.grad, np.full(3, 4 / 3))

    def test_constant_reductions_do_not_build_graph(self):
        for name in ("sum", "mean"):
            with self.subTest(name=name):
                y = getattr(Tensor([1., 2.]), name)()
                self.assertFalse(y.requires_grad)
                self.assertIsNone(y.grad_fn)

    def test_invalid_dimensions(self):
        for name in ("sum", "mean"):
            for dim, error in ((2, IndexError), ((0, 0), ValueError)):
                with self.subTest(name=name, dim=dim):
                    with self.assertRaises(error):
                        getattr(Tensor(np.ones((2, 3))), name)(dim=dim)


class ReductionIntegrationTests(unittest.TestCase):
    def test_operator_exports_and_keyword_metadata(self):
        from minitorch.ops import Mean, Sum
        for operator in (Sum, Mean):
            with self.subTest(operator=operator):
                x = Tensor([[1., 2.], [3., 4.]], requires_grad=True)
                y = operator.apply(x, axis=1, keepdims=True)
                self.assertEqual(y.data.shape, (2, 1))
                self.assertEqual(y.grad_fn.parents, (x,))
                self.assertEqual(y.grad_fn.ctx.needs_input_grad, (True,))
                y.backward([[2.], [4.]])
                np.testing.assert_allclose(x.grad, [[2., 2.], [4., 4.]] if operator is Sum else [[1., 1.], [2., 2.]])


if __name__ == "__main__":
    unittest.main()
