"""损失模块分层、交叉熵数值与梯度，以及完整分类训练。"""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minitorch import Tensor, nn, ops, optim
from minitorch.nn import functional as F


def reference_loss(logits, target, weight=None, reduction="mean",
                   ignore_index=-100, label_smoothing=0.):
    """逐样本参考公式；使用 logaddexp，独立于算子的向量化实现。"""
    axis = 0 if logits.ndim == 1 else 1
    rows = np.moveaxis(logits, axis, -1)
    classes = rows.shape[-1]
    weights = np.ones(classes) if weight is None else weight
    soft = target.shape == logits.shape
    targets = np.moveaxis(target, axis, -1) if soft else target
    losses = np.zeros(rows.shape[:-1])
    denominator = losses.size if soft else 0.
    for index in np.ndindex(losses.shape):
        if soft:
            distribution = targets[index]
        else:
            label = int(targets[index])
            if label == ignore_index:
                continue
            distribution = np.eye(classes)[label]
            denominator += weights[label]
        distribution = (1 - label_smoothing) * distribution + label_smoothing / classes
        log_probabilities = rows[index] - np.logaddexp.reduce(rows[index])
        losses[index] = -np.sum(distribution * weights * log_probabilities)
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        return losses.sum() / denominator


def numerical_gradient(objective, data):
    result = np.empty_like(data)
    for index in np.ndindex(data.shape):
        plus, minus = data.copy(), data.copy()
        plus[index] += 1e-6
        minus[index] -= 1e-6
        result[index] = (objective(plus) - objective(minus)) / 2e-6
    return result


class CrossEntropyTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(nn, "CrossEntropyLoss"), "尚未实现 nn.CrossEntropyLoss")

    def test_loss_inheritance_and_weight_is_a_buffer(self):
        from minitorch.nn.modules.loss import _Loss, _WeightedLoss

        weight = Tensor([1., 2., 3.])
        criterion = nn.CrossEntropyLoss(weight, reduction="sum")
        self.assertIsInstance(criterion, _WeightedLoss)
        self.assertIsInstance(criterion, _Loss)
        self.assertIsInstance(criterion, nn.Module)
        self.assertIs(nn.modules.CrossEntropyLoss, nn.CrossEntropyLoss)
        self.assertEqual(criterion.reduction, "sum")
        self.assertEqual(list(criterion.named_buffers()), [("weight", weight)])
        self.assertEqual(list(criterion.parameters()), [])
        self.assertIsNone(nn.CrossEntropyLoss().weight)
        with self.assertRaises(NotImplementedError):
            _Loss()(Tensor([1.]), Tensor(0.))

    def test_functional_module_and_operator_agree(self):
        x = Tensor([[1., 2., 3.], [-2., 1., 0.]], requires_grad=True)
        target = Tensor([2., 0.])
        criterion = nn.CrossEntropyLoss()
        module_loss = criterion(input=x, target=target)
        np.testing.assert_allclose(module_loss.data, F.cross_entropy(x, target).data)
        np.testing.assert_allclose(module_loss.data, ops.CrossEntropy.apply(x, target).data)
        module_loss.backward()
        self.assertEqual(x.grad.shape, x.shape)
        self.assertIsNone(target.grad)

    def test_class_index_forward_reductions(self):
        logits = Tensor(np.log([[1., 2., 1.], [3., 1., 2.]]))
        target = Tensor([1., 0.])
        for reduction, expected in (("none", [np.log(2), np.log(2)]),
                                    ("sum", 2 * np.log(2)), ("mean", np.log(2))):
            with self.subTest(reduction=reduction):
                np.testing.assert_allclose(F.cross_entropy(logits, target, reduction=reduction).data,
                                           expected)

    def test_weighted_mean_uses_nonignored_target_weight_sum(self):
        x = Tensor(np.zeros((3, 3)), requires_grad=True)
        target = Tensor([0., -100., 2.])
        weight = Tensor([1., 2., 4.])
        loss = F.cross_entropy(x, target, weight)
        np.testing.assert_allclose(loss.data, np.log(3))
        loss.backward()
        np.testing.assert_allclose(x.grad, [[-2 / 15, 1 / 15, 1 / 15],
                                            [0., 0., 0.], [4 / 15, 4 / 15, -8 / 15]])
        y = F.cross_entropy(x, target, weight, reduction="none")
        np.testing.assert_allclose(y.data, [np.log(3), 0., 4 * np.log(3)])

    def test_unbatched_and_spatial_class_axis(self):
        rng = np.random.default_rng(7)
        for shape in ((3,), (1, 3), (2, 3, 2), (2, 3, 2, 2)):
            data = rng.normal(size=shape)
            target_shape = () if len(shape) == 1 else (shape[0], *shape[2:])
            targets = rng.integers(0, 3, size=target_shape).astype(float)
            with self.subTest(shape=shape):
                loss = F.cross_entropy(Tensor(data), Tensor(targets), reduction="none")
                self.assertEqual(loss.shape, target_shape)
                np.testing.assert_allclose(loss.data, reference_loss(data, targets, reduction="none"))

    def test_weighted_smoothed_spatial_gradients_match_finite_differences(self):
        data = np.random.default_rng(3).normal(size=(2, 3, 2, 2))
        target = np.array([[[0., 1.], [2., -100.]], [[1., 2.], [0., 1.]]])
        weight = np.array([0.5, 2., 1.5])
        for reduction in ("none", "sum", "mean"):
            with self.subTest(reduction=reduction):
                seed = np.arange(1., 9.).reshape(target.shape) if reduction == "none" else 1.7
                options = dict(weight=weight, reduction=reduction, label_smoothing=0.2)
                expected = reference_loss(data, target, **options)
                x = Tensor(data, requires_grad=True)
                loss = F.cross_entropy(x, Tensor(target), Tensor(weight),
                                       reduction=reduction, label_smoothing=0.2)
                np.testing.assert_allclose(loss.data, expected)
                loss.backward(seed)
                numeric = numerical_gradient(
                    lambda values: np.sum(reference_loss(values, target, **options) * seed), data)
                np.testing.assert_allclose(x.grad, numeric, rtol=1e-6, atol=1e-7)

    def test_probability_target_input_target_and_weight_gradients(self):
        rng = np.random.default_rng(11)
        data = rng.normal(size=(2, 3, 2))
        targets = rng.uniform(size=data.shape)
        targets /= targets.sum(axis=1, keepdims=True)
        weights = np.array([1., 0.5, 2.])
        for reduction in ("none", "sum", "mean"):
            with self.subTest(reduction=reduction):
                seed = np.array([[1., 2.], [3., 4.]]) if reduction == "none" else 0.7
                x, target, weight = (Tensor(values, requires_grad=True)
                                     for values in (data, targets, weights))
                loss = nn.CrossEntropyLoss(weight, reduction=reduction, label_smoothing=0.3)(x, target)
                options = dict(reduction=reduction, label_smoothing=0.3)
                expected = reference_loss(data, targets, weights, **options)
                np.testing.assert_allclose(loss.data, expected)
                loss.backward(seed)
                objectives = (
                    lambda values: reference_loss(values, targets, weights, **options),
                    lambda values: reference_loss(data, values, weights, **options),
                    lambda values: reference_loss(data, targets, values, **options),
                )
                for tensor, values, objective in zip((x, target, weight), (data, targets, weights), objectives):
                    numeric = numerical_gradient(lambda array: np.sum(objective(array) * seed), values)
                    np.testing.assert_allclose(tensor.grad, numeric, rtol=1e-6, atol=1e-7)

    def test_probability_weighted_mean_divides_by_sample_count(self):
        x = Tensor(np.zeros((2, 3)))
        labels = Tensor([0., 2.])
        probabilities = Tensor([[1., 0., 0.], [0., 0., 1.]])
        weight = Tensor([1., 2., 4.])
        np.testing.assert_allclose(F.cross_entropy(x, labels, weight).data, np.log(3))
        np.testing.assert_allclose(F.cross_entropy(x, probabilities, weight).data, 2.5 * np.log(3))

    def test_probability_targets_do_not_require_normalization(self):
        data = np.array([2., -1., 0.])
        target = np.array([0.2, -0.1, 0.5])
        x = Tensor(data, requires_grad=True)
        loss = F.cross_entropy(x, Tensor(target), label_smoothing=1.)
        np.testing.assert_allclose(loss.data, reference_loss(data, target, label_smoothing=1.))
        loss.backward()
        expected = numerical_gradient(lambda values: reference_loss(values, target, label_smoothing=1.), data)
        np.testing.assert_allclose(x.grad, expected, atol=1e-8)

    def test_extreme_logits_are_finite_and_translation_invariant(self):
        data = np.array([[1000., -1000.], [-1000., 1000.]])
        target = Tensor([1., 1.])
        x = Tensor(data, requires_grad=True)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            loss = F.cross_entropy(x, target)
            loss.backward()
            translated = F.cross_entropy(Tensor(data + 10000), target)
        np.testing.assert_allclose(loss.data, 1000.)
        np.testing.assert_allclose(translated.data, loss.data)
        np.testing.assert_allclose(x.grad, [[0.5, -0.5], [0., 0.]])

    def test_masked_classes_do_not_poison_index_loss(self):
        x = Tensor([[0., -np.inf], [-np.inf, 0.]], requires_grad=True)
        with np.errstate(invalid="raise"):
            loss = F.cross_entropy(x, Tensor([0., 1.]))
            loss.backward()
        np.testing.assert_array_equal(loss.data, 0.)
        np.testing.assert_array_equal(x.grad, np.zeros((2, 2)))

    def test_unused_class_weights_do_not_poison_index_gradients(self):
        for unused_weight in (np.inf, np.nan):
            for reduction in ("none", "sum", "mean"):
                with self.subTest(weight=unused_weight, reduction=reduction):
                    x = Tensor([[0., 0.], [2., 3.]], requires_grad=True)
                    with np.errstate(invalid="raise"):
                        loss = F.cross_entropy(x, Tensor([0., -100.]), Tensor([1., unused_weight]),
                                               reduction=reduction)
                        loss.backward([2., 9.] if reduction == "none" else 2.)
                    expected = [np.log(2), 0.] if reduction == "none" else np.log(2)
                    np.testing.assert_allclose(loss.data, expected)
                    np.testing.assert_allclose(x.grad, [[-1., 1.], [0., 0.]])

    def test_all_ignored_targets_have_zero_gradients(self):
        for smoothing in (0., 0.2):
            for reduction in ("none", "sum", "mean"):
                with self.subTest(smoothing=smoothing, reduction=reduction):
                    x = Tensor(np.ones((2, 3)), requires_grad=True)
                    loss = F.cross_entropy(x, Tensor([-7., -7.]), ignore_index=-7,
                                           reduction=reduction, label_smoothing=smoothing)
                    if reduction == "mean":
                        self.assertTrue(np.isnan(loss.data))
                    else:
                        np.testing.assert_array_equal(loss.data, [0., 0.] if reduction == "none" else 0.)
                    loss.backward(np.ones(2) if reduction == "none" else None)
                    np.testing.assert_array_equal(x.grad, np.zeros((2, 3)))

    def test_empty_batches_and_spatial_dimensions(self):
        for shape in ((0, 3), (2, 3, 0)):
            for soft in (False, True):
                for reduction in ("none", "sum", "mean"):
                    with self.subTest(shape=shape, soft=soft, reduction=reduction):
                        x = Tensor(np.empty(shape), requires_grad=True)
                        target_shape = shape if soft else (shape[0], *shape[2:])
                        target = Tensor(np.empty(target_shape), requires_grad=soft)
                        weight = Tensor([1., 2., 3.], requires_grad=soft)
                        loss = F.cross_entropy(x, target, weight, reduction=reduction)
                        if reduction == "mean":
                            self.assertTrue(np.isnan(loss.data))
                        elif reduction == "sum":
                            np.testing.assert_array_equal(loss.data, 0.)
                        else:
                            self.assertEqual(loss.shape, (shape[0], *shape[2:]))
                        loss.backward(np.empty(loss.shape) if reduction == "none" else None)
                        self.assertEqual(x.grad.shape, shape)
                        if soft:
                            self.assertEqual(target.grad.shape, target_shape)
                            np.testing.assert_array_equal(weight.grad, np.zeros(3))

    def test_zero_class_weights_keep_nan_weighted_mean(self):
        x = Tensor([[1., 2., 3.]], requires_grad=True)
        loss = F.cross_entropy(x, Tensor([1.]), Tensor([0., 0., 0.]))
        self.assertTrue(np.isnan(loss.data))
        loss.backward()
        self.assertTrue(np.isnan(x.grad).all())

    def test_backward_uses_forward_snapshots(self):
        data = np.array([[0.2, 0.7, -0.1], [-0.3, 0.4, 0.8]])
        targets = np.array([[0.2, 0.3, 0.5], [0.1, 0.7, 0.2]])
        weights = np.array([1., 2., 0.5])
        x, target, weight = (Tensor(values, requires_grad=True) for values in (data, targets, weights))
        loss = F.cross_entropy(x, target, weight)
        x.data[:] = 100
        target.data[:] = 0
        weight.data[:] = 9
        loss.backward()
        numeric = numerical_gradient(lambda values: reference_loss(values, targets, weights), data)
        np.testing.assert_allclose(x.grad, numeric, rtol=1e-6, atol=1e-7)
        numeric_target = numerical_gradient(lambda values: reference_loss(data, values, weights), targets)
        np.testing.assert_allclose(target.grad, numeric_target, rtol=1e-6, atol=1e-7)

    def test_constants_skip_graph_and_shared_loss_accumulates(self):
        target = Tensor([1.])
        constant = F.cross_entropy(Tensor([[0., 0.]]), target)
        self.assertFalse(constant.requires_grad)
        self.assertIsNone(constant.grad_fn)
        x = Tensor([[0., 0.]], requires_grad=True)
        loss = F.cross_entropy(x, target)
        shared = loss + 2 * loss
        shared.backward()
        shared.backward()
        np.testing.assert_allclose(x.grad, [[3., -3.]])

    def test_legacy_arguments_override_reduction(self):
        x, target = Tensor([[0., 0.], [0., 0.]]), Tensor([0., 1.])
        with self.assertWarns(UserWarning):
            criterion = nn.CrossEntropyLoss(size_average=False, reduction="none")
        self.assertEqual(criterion.reduction, "sum")
        np.testing.assert_allclose(criterion(x, target).data, 2 * np.log(2))
        with self.assertWarns(UserWarning):
            losses = F.cross_entropy(x, target, reduce=False, reduction="sum")
        np.testing.assert_allclose(losses.data, [np.log(2), np.log(2)])

    def test_invalid_arguments_are_rejected(self):
        x, target = Tensor([[1., 2., 3.]]), Tensor([1.])
        for name, value in (("input", [[1., 2., 3.]]), ("target", [1.]),
                            ("weight", [1., 2., 3.])):
            arguments = dict(input=x, target=target)
            arguments[name] = value
            with self.subTest(name=name), self.assertRaises(TypeError):
                F.cross_entropy(**arguments)
        for smoothing in (-0.1, 1.1, np.nan, np.inf):
            with self.subTest(smoothing=smoothing), self.assertRaises(ValueError):
                F.cross_entropy(x, target, label_smoothing=smoothing)
        for smoothing in (True, "0.1"):
            with self.assertRaises(TypeError):
                F.cross_entropy(x, target, label_smoothing=smoothing)
        for ignore_index in (True, 0.5, "-100"):
            with self.assertRaises(TypeError):
                F.cross_entropy(x, target, ignore_index=ignore_index)
        with self.assertRaises(ValueError):
            F.cross_entropy(x, target, reduction="median")
        with self.assertRaises(ValueError):
            nn.CrossEntropyLoss(reduction="median")
        with self.assertRaises(TypeError):
            nn.CrossEntropyLoss(weight=[1., 2., 3.])

    def test_invalid_shapes_labels_and_differentiability(self):
        x = Tensor([[1., 2., 3.]])
        for input, target in ((Tensor(1.), Tensor(0.)), (Tensor(np.empty((2, 0))), Tensor([0., 0.])),
                              (x, Tensor([[1.]])), (x, Tensor([0., 1.]))):
            with self.subTest(input=input.shape, target=target.shape), self.assertRaises(ValueError):
                F.cross_entropy(input, target)
        for label in (0.5, np.nan, np.inf):
            with self.subTest(label=label), self.assertRaises(ValueError):
                F.cross_entropy(x, Tensor([label]))
        for label in (-1., 3., 1e30):
            with self.subTest(label=label), self.assertRaises(IndexError):
                F.cross_entropy(x, Tensor([label]))
        for weight in (Tensor([1., 2.]), Tensor([[1., 2., 3.]])):
            with self.assertRaises(ValueError):
                F.cross_entropy(x, Tensor([1.]), weight)
        with self.assertRaises(RuntimeError):
            F.cross_entropy(x, Tensor([1.], requires_grad=True))
        with self.assertRaises(RuntimeError):
            F.cross_entropy(x, Tensor([1.]), Tensor([1., 2., 3.], requires_grad=True))
        with self.assertRaises(ValueError):
            F.cross_entropy(x, Tensor([[0., 1., 0.]]), ignore_index=1)

    def test_linear_classifier_training_converges(self):
        np.random.seed(5)
        x = Tensor([[-2., -1.], [-1., -2.], [2., -1.], [1., -2.], [0., 2.], [1., 3.]])
        target = Tensor([0., 0., 1., 1., 2., 2.])
        model = nn.Linear(2, 3)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.2)
        initial = float(criterion(model(x), target).data)
        for _ in range(100):
            optimizer.zero_grad()
            loss = criterion(model(x), target)
            loss.backward()
            optimizer.step()
        logits = model(x)
        final = float(criterion(logits, target).data)
        self.assertLess(final, 0.1 * initial)
        np.testing.assert_array_equal(logits.data.argmax(axis=1), target.data)
        for parameter in model.parameters():
            self.assertTrue(np.isfinite(parameter.grad).all())


if __name__ == "__main__":
    unittest.main()
