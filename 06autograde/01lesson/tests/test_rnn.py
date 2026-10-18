"""单层单向 RNN 的公开接口、序列输出与 BPTT。"""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minitorch import Tensor, nn
from minitorch.nn.modules.rnn import RNN


class RNNTests(unittest.TestCase):
    def scalar_model(self, **kwargs):
        model = RNN(input_size=1, hidden_size=1, **kwargs)
        model.weight_ih_l0.data[:] = 1.
        model.weight_hh_l0.data[:] = 1.
        if model.bias:
            model.bias_ih_l0.data[:] = 0.
            model.bias_hh_l0.data[:] = 0.
        return model

    def test_export_and_default_hidden_state_return_all_steps(self):
        self.assertTrue(hasattr(nn, "RNN"), "缺少 nn.RNN")
        self.assertIs(nn.RNN, RNN)
        model = self.scalar_model()
        output, h_n = model(input=Tensor([[[1.], [0.]], [[1.], [1.]]]))
        self.assertEqual(output.shape, (2, 2, 1))
        self.assertEqual(h_n.shape, (1, 2, 1))
        np.testing.assert_allclose(output.data, [
            [[0.7615941559557649], [0.]],
            [[0.9426807890983486], [0.7615941559557649]],
        ])
        np.testing.assert_allclose(h_n.data, output.data[-1:])

    def test_explicit_hidden_state_and_batch_first(self):
        model = self.scalar_model(batch_first=True)
        h0 = Tensor([[[0.], [1.]]], requires_grad=True)
        output, h_n = model(input=Tensor([[[1.], [1.]], [[0.], [0.]]]), hx=h0)
        self.assertEqual(output.shape, (2, 2, 1))
        self.assertEqual(h_n.shape, (1, 2, 1))
        np.testing.assert_allclose(output.data, [
            [[0.7615941559557649], [0.9426807890983486]],
            [[0.7615941559557649], [0.6420149920119997]],
        ])
        np.testing.assert_array_equal(h0.data, [[[0.], [1.]]])

    def test_unbatched_input_ignores_batch_first(self):
        for batch_first in (False, True):
            with self.subTest(batch_first=batch_first):
                model = self.scalar_model(batch_first=batch_first)
                for hx in (None, Tensor([[0.]])):
                    output, h_n = model(Tensor([[1.], [1.]]), hx)
                    self.assertEqual(output.shape, (2, 1))
                    self.assertEqual(h_n.shape, (1, 1))
                    np.testing.assert_allclose(output.data, [[0.7615941559557649], [0.9426807890983486]])

    def test_relu_and_disabled_bias(self):
        model = self.scalar_model(bias=False, nonlinearity="relu")
        output, h_n = model(Tensor([[[-1.]], [[2.]]]))
        np.testing.assert_array_equal(output.data, [[[0.]], [[2.]]])
        np.testing.assert_array_equal(h_n.data, [[[2.]]])
        self.assertIsNone(model.bias_ih_l0)
        self.assertIsNone(model.bias_hh_l0)
        self.assertEqual(len(list(model.parameters())), 2)

    def test_parameter_layout_and_initialization(self):
        model = RNN(input_size=np.int64(3), hidden_size=2)
        params = dict(model.named_parameters())
        self.assertEqual({name: p.shape for name, p in params.items()}, {
            "weight_ih_l0": (2, 3), "weight_hh_l0": (2, 2),
            "bias_ih_l0": (2,), "bias_hh_l0": (2,),
        })
        for p in params.values():
            self.assertIsInstance(p, nn.Parameter)
            self.assertTrue(p.requires_grad)
            self.assertTrue(np.isfinite(p.data).all())
            self.assertTrue((np.abs(p.data) <= 1 / np.sqrt(2)).all())

    def test_invalid_dimensions_and_unsupported_layers(self):
        for dims in ((0, 2), (2, 0), (-1, 2), (2, -1)):
            with self.subTest(dims=dims), self.assertRaises(ValueError):
                RNN(*dims)
        for dims in ((True, 2), (2, False), (2.5, 2), (2, "3")):
            with self.subTest(dims=dims), self.assertRaises(TypeError):
                RNN(*dims)
        with self.assertRaises(ValueError):
            RNN(2, 3, num_layers=0)
        with self.assertRaises(NotImplementedError):
            RNN(2, 3, num_layers=2)
        with self.assertRaises(ValueError):
            RNN(2, 3, nonlinearity="sigmoid")

    def test_rejects_invalid_input_and_hidden_shapes(self):
        model = RNN(2, 3)
        for shape in ((), (2,), (2, 1, 1), (2, 1, 1, 2), (0, 4, 2)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                model(Tensor(np.zeros(shape)))
        x = Tensor(np.zeros((2, 4, 2)))
        for shape in ((4, 3), (1, 1, 3), (1, 4, 1), (2, 4, 3)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                model(x, Tensor(np.zeros(shape)))
        with self.assertRaises(ValueError):
            model(Tensor(np.zeros((2, 2))), Tensor(np.zeros((1, 1, 3))))
        with self.assertRaises(TypeError):
            model([Tensor(np.zeros((4, 2)))])
        with self.assertRaises(TypeError):
            model(x, np.zeros((1, 4, 3)))

    def test_bptt_for_outputs_final_state_inputs_and_all_parameters(self):
        # Independent NumPy recurrence checks weight orientation and both biases.
        model = RNN(2, 2)
        model.weight_ih_l0.data[:] = [[0.2, -0.3], [0.4, 0.1]]
        model.weight_hh_l0.data[:] = [[0.1, 0.2], [-0.2, 0.3]]
        model.bias_ih_l0.data[:] = [0.05, -0.1]
        model.bias_hh_l0.data[:] = [-0.02, 0.03]
        x = Tensor([[[1., -0.5], [0.2, 0.7]],
                    [[0.3, 0.4], [-0.6, 0.8]],
                    [[0.1, 0.2], [-0.3, 0.5]]], requires_grad=True)
        hx = Tensor([[[0.1, -0.2], [0.3, 0.4]]], requires_grad=True)

        def reference():
            state = hx.data[0].copy()
            outputs = []
            for step in x.data:
                state = np.tanh(step @ model.weight_ih_l0.data.T
                                + state @ model.weight_hh_l0.data.T
                                + model.bias_ih_l0.data + model.bias_hh_l0.data)
                outputs.append(state)
            return np.stack(outputs), state[None]

        output, h_n = model(x, hx)
        expected_output, expected_h = reference()
        np.testing.assert_allclose(output.data, expected_output)
        np.testing.assert_allclose(h_n.data, expected_h)
        ((output ** 2).sum() + 2 * (h_n ** 2).sum()).backward()
        for tensor in [x, hx, *model.parameters()]:
            numeric = np.empty_like(tensor.data)
            for index in np.ndindex(tensor.shape):
                original = tensor.data[index]
                objectives = []
                for delta in (1e-6, -1e-6):
                    tensor.data[index] = original + delta
                    out, final = reference()
                    objectives.append((out ** 2).sum() + 2 * (final ** 2).sum())
                tensor.data[index] = original
                numeric[index] = (objectives[0] - objectives[1]) / 2e-6
            np.testing.assert_allclose(tensor.grad, numeric, rtol=1e-5, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
