"""职责拆分的行为回归测试；只依赖 NumPy 和标准库。"""

import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

LESSON_DIR = Path(__file__).resolve().parents[1]
LEGACY_FILE = LESSON_DIR / "minitorch/autograd/02my_stu_autograde.py"
sys.path.insert(0, str(LESSON_DIR))

from minitorch import Tensor


class AutogradBehaviorTests(unittest.TestCase):
    def test_shared_input_and_shared_intermediate(self):
        x = Tensor(3.0, requires_grad=True)
        u = x * x
        y = u * u + u
        y.backward()
        np.testing.assert_allclose(y.data, 90.0)
        np.testing.assert_allclose(x.grad, 114.0)
        self.assertIsNone(u.grad)
        self.assertIsNone(y.grad)

    def test_arithmetic_and_reverse_operators(self):
        expressions = (
            (lambda x: x + 2, 5, 1),
            (lambda x: 2 + x, 5, 1),
            (lambda x: x - 2, 1, 1),
            (lambda x: 2 - x, -1, -1),
            (lambda x: -x, -3, -1),
            (lambda x: x * 2, 6, 2),
            (lambda x: 2 * x, 6, 2),
            (lambda x: x / 2, 1.5, 0.5),
            (lambda x: 6 / x, 2, -2 / 3),
            (lambda x: x ** 2, 9, 6),
            (lambda x: 2 ** x, 8, 8 * np.log(2)),
        )
        for expression, value, gradient in expressions:
            with self.subTest(value=value, gradient=gradient):
                x = Tensor(3.0, requires_grad=True)
                y = expression(x)
                y.backward()
                np.testing.assert_allclose(y.data, value)
                np.testing.assert_allclose(x.grad, gradient)

    def test_broadcast_add_and_multiply(self):
        a = Tensor([[1.0], [2.0]], requires_grad=True)
        b = Tensor([[3.0, 4.0, 5.0]], requires_grad=True)
        y = a * b + a
        weights = np.array([[1., 2., 3.], [4., 5., 6.]])
        y.backward(weights)
        np.testing.assert_allclose(a.grad, (weights * (b.data + 1)).sum(1, keepdims=True))
        np.testing.assert_allclose(b.grad, (weights * a.data).sum(0, keepdims=True))

    def test_scalar_broadcast_and_constant_input(self):
        x = Tensor(2.0, requires_grad=True)
        constant = Tensor([1.0, 2.0, 3.0])
        (x * constant).backward(np.ones(3))
        np.testing.assert_allclose(x.grad, 6.0)
        self.assertEqual(x.grad.shape, ())
        self.assertIsNone(constant.grad)

    def test_power_exponent_gradient(self):
        a = Tensor([2., 3.], requires_grad=True)
        b = Tensor(2.0, requires_grad=True)
        (a ** b).backward(np.ones(2))
        np.testing.assert_allclose(a.grad, [4., 6.])
        np.testing.assert_allclose(b.grad, np.sum(a.data ** 2 * np.log(a.data)))

    def test_matmul_and_batch_broadcast(self):
        a = Tensor(np.arange(12.).reshape(2, 2, 3), requires_grad=True)
        b = Tensor(np.arange(12.).reshape(3, 4), requires_grad=True)
        y = a @ b
        weights = np.arange(16.).reshape(2, 2, 4)
        y.backward(weights)
        np.testing.assert_allclose(y.data, np.matmul(a.data, b.data))
        np.testing.assert_allclose(a.grad, weights @ b.data.T)
        np.testing.assert_allclose(b.grad, (np.swapaxes(a.data, -1, -2) @ weights).sum(0))

    def test_reverse_matmul(self):
        x = Tensor([[1., 2.], [3., 4.]], requires_grad=True)
        y = [[2., 0.]] @ x
        y.backward([[1., 3.]])
        np.testing.assert_allclose(y.data, [[2., 4.]])
        np.testing.assert_allclose(x.grad, [[2., 6.], [0., 0.]])

    def test_repeated_backward_and_zero_grad(self):
        x = Tensor(3.0, requires_grad=True)
        y = x * x
        y.backward()
        y.backward()
        np.testing.assert_allclose(x.grad, 12.0)
        x.zero_grad()
        self.assertIsNone(x.grad)
        y.backward()
        np.testing.assert_allclose(x.grad, 6.0)

    def test_saved_values_are_copied(self):
        x = Tensor(3.0, requires_grad=True)
        y = x * x
        x.data[...] = 100.0
        y.backward()
        np.testing.assert_allclose(x.grad, 6.0)

    def test_leaf_gradient_does_not_alias_seed(self):
        x = Tensor([1., 2.], requires_grad=True)
        seed = np.array([3., 4.])
        x.backward(seed)
        seed[:] = 0
        np.testing.assert_allclose(x.grad, [3., 4.])

    def test_long_chain_uses_iterative_traversal(self):
        x = Tensor(1.0, requires_grad=True)
        y = x
        for _ in range(1500):
            y = y + 1
        y.backward()
        np.testing.assert_allclose(x.grad, 1.0)

    def test_invalid_backward_inputs(self):
        with self.assertRaisesRegex(RuntimeError, "does not require gradients"):
            Tensor(1.0).backward()
        y = Tensor([1., 2.], requires_grad=True) * 2
        with self.assertRaisesRegex(RuntimeError, "多元素"):
            y.backward()
        with self.assertRaisesRegex(ValueError, "gradient 形状"):
            y.backward([1.])

    def test_operator_constraints(self):
        x = Tensor(-2.0, requires_grad=True)
        (x ** 2).backward()
        np.testing.assert_allclose(x.grad, -4.)
        with self.assertRaisesRegex(ValueError, "底数必须大于 0"):
            (x ** Tensor(2.0, requires_grad=True)).backward()
        with self.assertRaisesRegex(ValueError, "至少为二维"):
            Tensor([1., 2.]) @ Tensor([3., 4.])


class PackageIntegrationTests(unittest.TestCase):
    def test_public_backward(self):
        from minitorch.autograd import backward
        x = Tensor(3.0, requires_grad=True)
        backward(x * 2)
        np.testing.assert_allclose(x.grad, 2.0)

    def test_original_and_package_match_behavior(self):
        spec = importlib.util.spec_from_file_location("legacy_autograd", LEGACY_FILE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for tensor_class in (module.Tensor, Tensor):
            with self.subTest(tensor_class=tensor_class):
                x = tensor_class(3.0, requires_grad=True)
                y = x * x + 2 * x
                y.backward()
                np.testing.assert_allclose(y.data, 15.0)
                np.testing.assert_allclose(x.grad, 8.0)

    def test_entry_runs_from_another_directory(self):
        result = subprocess.run(
            [sys.executable, str(LEGACY_FILE)], cwd=LESSON_DIR.parent,
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_independent_module_imports(self):
        for module in ("tensor", "ops.arithmetic", "ops.linalg", "autograd.function", "autograd.graph", "autograd.engine"):
            with self.subTest(module=module):
                result = subprocess.run(
                    [sys.executable, "-c", f"import minitorch.{module}"],
                    cwd=LESSON_DIR, capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_custom_function_and_gradient_validation(self):
        from minitorch.autograd import Function

        class Square(Function):
            @staticmethod
            def forward(ctx, a):
                ctx.save_for_backward(a)
                return a * a

            @staticmethod
            def backward(ctx, gradient):
                return (2 * ctx.saved_values[0] * gradient,)

        x = Tensor(3.0, requires_grad=True)
        Square.apply(x).backward()
        np.testing.assert_allclose(x.grad, 6.0)

        class WrongCount(Square):
            @staticmethod
            def backward(ctx, gradient):
                return gradient, gradient

        class WrongShape(Square):
            @staticmethod
            def backward(ctx, gradient):
                return (np.ones(2),)

        with self.assertRaisesRegex(RuntimeError, "expected 1"):
            WrongCount.apply(x).backward()
        with self.assertRaisesRegex(RuntimeError, "返回梯度形状"):
            WrongShape.apply(x).backward()


if __name__ == "__main__":
    unittest.main()
