"""网络基类与现有 Tensor/自动求导的集成测试。"""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minitorch import Tensor, nn
from minitorch.nn import Module, Parameter


class Affine(Module):
    def __init__(self):
        super().__init__()
        self.weight = Parameter([[1.], [2.]])
        self.bias = Parameter([0.5])

    def forward(self, x, *, scale=1):
        return (x @ self.weight + self.bias) * scale


class ParameterTests(unittest.TestCase):
    def test_parameter_participates_in_autograd(self):
        p = Parameter([2., 3.])
        self.assertIsInstance(p, Tensor)
        self.assertTrue(p.requires_grad)
        self.assertIsNone(p.grad_fn)
        result = (p * p).sum()
        self.assertIs(type(result), Tensor)
        result.backward()
        np.testing.assert_allclose(p.grad, [4., 6.])

    def test_tensor_input_becomes_independent_leaf(self):
        source = Tensor([2., 3.], requires_grad=True)
        intermediate = source * 2
        p = Parameter(intermediate)
        intermediate.data[:] = 100
        np.testing.assert_allclose(p.data, [4., 6.])
        self.assertIsNone(p.grad_fn)
        p.sum().backward()
        np.testing.assert_allclose(p.grad, [1., 1.])
        self.assertIsNone(source.grad)
        self.assertIsNone(intermediate.grad)

    def test_frozen_parameter_is_registered_but_receives_no_gradient(self):
        model = Module()
        model.frozen = Parameter([2., 3.], requires_grad=False)
        model.weight = Parameter([1., 2.])
        (model.frozen * model.weight).sum().backward()
        self.assertEqual(list(model.parameters()), [model.frozen, model.weight])
        self.assertIsNone(model.frozen.grad)
        np.testing.assert_allclose(model.weight.grad, [2., 3.])

    def test_empty_parameter(self):
        p = Parameter()
        self.assertEqual(p.data.shape, (0,))
        self.assertTrue(p.requires_grad)


class ModuleTests(unittest.TestCase):
    def test_forward_backward_and_parameter_collection(self):
        model = Affine()
        x = Tensor([[1., 2.], [3., 4.]])
        y = model(x)
        np.testing.assert_allclose(y.data, [[5.5], [11.5]])
        loss = (y ** 2).mean()
        loss.backward()
        np.testing.assert_allclose(model.weight.grad, [[40.], [57.]])
        np.testing.assert_allclose(model.bias.grad, [17.])
        self.assertEqual(list(model.named_parameters()), [("weight", model.weight), ("bias", model.bias)])

    def test_call_preserves_keyword_arguments(self):
        model = Affine()
        np.testing.assert_allclose(model(Tensor([[1., 2.]]), scale=2).data, [[11.]])

    def test_nested_parameters_and_modules(self):
        model = Module()
        model.offset = Parameter(1.)
        model.layer = Affine()
        model.layer.extra = Module()
        model.layer.extra.factor = Parameter(2.)
        names = [name for name, _ in model.named_parameters(prefix="net")]
        self.assertEqual(names, ["net.offset", "net.layer.weight", "net.layer.bias", "net.layer.extra.factor"])
        self.assertEqual(list(model.parameters(recurse=False)), [model.offset])
        self.assertEqual([name for name, _ in model.named_modules()], ["", "layer", "layer.extra"])
        self.assertEqual(list(model.children()), [model.layer])

    def test_plain_tensors_and_python_containers_are_not_registered(self):
        model = Module()
        model.cache = Tensor(1., requires_grad=True)
        model.items = [Parameter(2.), Affine()]
        model.config = {"weight": Parameter(3.)}
        self.assertEqual(list(model.parameters()), [])
        self.assertEqual(list(model.children()), [])

    def test_shared_parameter_is_deduplicated(self):
        model = Module()
        p = Parameter(2.)
        model.weight = p
        model.alias = p
        model.layer = Module()
        model.layer.shared = p
        self.assertEqual(list(model.named_parameters()), [("weight", p)])
        self.assertEqual([name for name, _ in model.named_parameters(remove_duplicate=False)], ["weight", "alias", "layer.shared"])
        (model.weight * model.alias + model.layer.shared).backward()
        np.testing.assert_allclose(p.grad, 5.)

    def test_shared_submodule_is_deduplicated(self):
        model = Module()
        shared = Affine()
        model.first = shared
        model.second = shared
        self.assertEqual(list(model.children()), [shared])
        self.assertEqual(list(model.modules()), [model, shared])
        self.assertEqual([name for name, _ in model.named_parameters()], ["first.weight", "first.bias"])
        self.assertEqual([name for name, _ in model.named_parameters(remove_duplicate=False)], ["first.weight", "first.bias", "second.weight", "second.bias"])

    def test_parameter_replacement_none_and_deletion(self):
        model = Module()
        original = Parameter(1.)
        model.weight = original
        replacement = Parameter(2.)
        model.weight = replacement
        self.assertEqual(list(model.parameters()), [replacement])
        model.weight = None
        self.assertIsNone(model.weight)
        self.assertEqual(list(model.parameters()), [])
        with self.assertRaises(TypeError):
            model.weight = Tensor(3.)
        self.assertIsNone(model.weight)
        model.weight = replacement
        del model.weight
        self.assertFalse(hasattr(model, "weight"))
        self.assertEqual(list(model.parameters()), [])
        model.weight = "ordinary attribute"
        self.assertEqual(model.weight, "ordinary attribute")

    def test_submodule_replacement_none_and_deletion(self):
        model = Module()
        model.layer = Affine()
        replacement = Affine()
        model.layer = replacement
        self.assertEqual(list(model.children()), [replacement])
        model.layer = None
        self.assertIsNone(model.layer)
        self.assertEqual(list(model.parameters()), [])
        with self.assertRaises(TypeError):
            model.layer = 1
        model.layer = replacement
        del model.layer
        self.assertEqual(list(model.children()), [])

    def test_ordinary_attribute_can_become_parameter_or_module(self):
        model = Module()
        model.value = 1
        model.value = Parameter(2.)
        self.assertEqual(list(model.parameters()), [model.value])
        model.layer = "placeholder"
        model.layer = Affine()
        self.assertEqual(list(model.children()), [model.layer])

    def test_submodule_can_be_replaced_by_parameter(self):
        model = Module()
        model.layer = Affine()
        model.layer = Parameter(2.)
        self.assertEqual(list(model.parameters()), [model.layer])
        self.assertEqual(list(model.children()), [])
        with self.assertRaises(TypeError):
            model.layer = Affine()

    def test_explicit_registration(self):
        model = Module()
        p = Parameter(2.)
        layer = Affine()
        model.register_parameter("weight", p)
        model.register_parameter("bias", None)
        model.add_module("layer", layer)
        self.assertIs(model.weight, p)
        self.assertIs(model.layer, layer)
        self.assertIsNone(model.bias)
        self.assertEqual([name for name, _ in model.named_parameters()], ["weight", "layer.weight", "layer.bias"])

    def test_registration_validation_keeps_existing_state(self):
        model = Affine()
        old_weight = model.weight
        for name, exception in (("", KeyError), ("a.b", KeyError), (1, TypeError), ("forward", KeyError), ("training", KeyError)):
            with self.subTest(name=name):
                with self.assertRaises(exception):
                    model.register_parameter(name, Parameter(1.))
                with self.assertRaises(exception):
                    model.add_module(name, Module())
        with self.assertRaises(TypeError):
            model.register_parameter("weight", Tensor(1.))
        with self.assertRaises(TypeError):
            model.add_module("layer", Tensor(1.))
        with self.assertRaises(KeyError):
            model.add_module("weight", Module())
        self.assertIs(model.weight, old_weight)

    def test_super_init_is_required(self):
        class BadParameter(Module):
            def __init__(self):
                self.weight = Parameter(1.)

        class BadModule(Module):
            def __init__(self):
                self.layer = Affine()

        for cls in (BadParameter, BadModule):
            with self.subTest(cls=cls):
                with self.assertRaises(AttributeError):
                    cls()

    def test_cycles_are_rejected_without_damaging_module(self):
        model = Module()
        child = Module()
        model.child = child
        with self.assertRaises(ValueError):
            model.self_reference = model
        with self.assertRaises(ValueError):
            child.parent = model
        self.assertEqual(list(model.modules()), [model, child])

    def test_train_eval_recurses_without_disabling_gradients(self):
        model = Module()
        model.layer = Affine()
        self.assertIs(model.eval(), model)
        self.assertFalse(model.training)
        self.assertFalse(model.layer.training)
        model.layer(Tensor([[1., 2.]])).sum().backward()
        np.testing.assert_allclose(model.layer.weight.grad, [[1.], [2.]])
        self.assertIs(model.train(), model)
        self.assertTrue(model.training)
        self.assertTrue(model.layer.training)
        with self.assertRaises(ValueError):
            model.train("eval")

    def test_zero_grad_defaults_to_none_and_supports_in_place_zero(self):
        model = Module()
        model.layer = Affine()
        model.unused = Parameter(1.)
        model.layer(Tensor([[1., 2.]])).sum().backward()
        gradient = model.layer.weight.grad
        model.zero_grad(set_to_none=False)
        self.assertIs(model.layer.weight.grad, gradient)
        np.testing.assert_allclose(gradient, [[0.], [0.]])
        self.assertIsNone(model.unused.grad)
        model.layer(Tensor([[1., 2.]])).sum().backward()
        np.testing.assert_allclose(model.layer.weight.grad, [[1.], [2.]])
        model.zero_grad()
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_forward_must_be_implemented(self):
        with self.assertRaises(NotImplementedError):
            Module()(Tensor(1.))

    def test_nn_exports(self):
        self.assertIs(nn.Parameter, Parameter)
        self.assertIs(nn.Module, Module)


if __name__ == "__main__":
    unittest.main()
