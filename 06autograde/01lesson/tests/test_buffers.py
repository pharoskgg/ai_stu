"""Module buffer 注册、递归遍历和替换行为。"""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minitorch import Tensor, nn


class BufferTests(unittest.TestCase):
    def test_buffers_are_attributes_but_not_parameters(self):
        module = nn.Module()
        value = Tensor([1., 2.])
        module.register_buffer("weight", value)
        module.register_buffer("optional", None)
        module.plain = Tensor(3.)
        self.assertIs(module.weight, value)
        self.assertIsNone(module.optional)
        self.assertEqual(list(module.named_buffers()), [("weight", value)])
        self.assertEqual(list(module.parameters()), [])

    def test_recursive_names_and_shared_buffers_are_deduplicated(self):
        module = nn.Module()
        shared = Tensor([2.])
        module.register_buffer("local", shared)
        module.child = nn.Module()
        module.child.register_buffer("alias", shared)
        module.child.register_buffer("other", Tensor(3.))
        module.alias = module.child
        self.assertEqual([name for name, _ in module.named_buffers(prefix="net")],
                         ["net.local", "net.child.other"])
        self.assertEqual(list(module.buffers(recurse=False)), [shared])
        self.assertEqual([name for name, _ in module.named_buffers(remove_duplicate=False)],
                         ["local", "child.alias", "child.other", "alias.alias", "alias.other"])

    def test_assignment_none_and_deletion_preserve_registration(self):
        module = nn.Module()
        module.register_buffer("cache", Tensor(1.), persistent=False)
        replacement = Tensor(2.)
        module.cache = replacement
        self.assertEqual(list(module.buffers()), [replacement])
        self.assertIn("cache", module._non_persistent_buffers_set)
        with self.assertRaises(TypeError):
            module.cache = 3
        self.assertIs(module.cache, replacement)
        module.cache = None
        self.assertEqual(list(module.buffers()), [])
        module.register_buffer("cache", replacement, persistent=True)
        self.assertNotIn("cache", module._non_persistent_buffers_set)
        del module.cache
        self.assertFalse(hasattr(module, "cache"))
        self.assertEqual(list(module.buffers()), [])
        self.assertNotIn("cache", module._non_persistent_buffers_set)

    def test_parameter_or_module_replaces_buffer_without_stale_entries(self):
        for replacement in (nn.Parameter(2.), nn.Module()):
            with self.subTest(replacement=type(replacement).__name__):
                module = nn.Module()
                module.register_buffer("value", Tensor(1.), persistent=False)
                module.value = replacement
                self.assertIs(module.value, replacement)
                self.assertEqual(list(module.buffers()), [])
                self.assertNotIn("value", module._non_persistent_buffers_set)
                expected_parameters = [replacement] if isinstance(replacement, nn.Parameter) else []
                self.assertEqual(list(module.parameters()), expected_parameters)

    def test_registration_errors_preserve_existing_state(self):
        module = nn.Module()
        original = Tensor(1.)
        module.register_buffer("value", original)
        for name, error in (("", KeyError), ("a.b", KeyError), (3, TypeError),
                            ("forward", KeyError), ("training", KeyError),
                            ("_buffers", KeyError), ("_non_persistent_buffers_set", KeyError)):
            with self.subTest(name=name):
                with self.assertRaises(error):
                    module.register_buffer(name, original)
        with self.assertRaises(TypeError):
            module.register_buffer("value", [2.])
        with self.assertRaises(TypeError):
            module.register_buffer("value", original, persistent="yes")
        module.param = nn.Parameter(2.)
        module.layer = nn.Module()
        module.plain = 3
        for name in ("param", "layer", "plain"):
            with self.assertRaises(KeyError):
                module.register_buffer(name, original)
        with self.assertRaises(KeyError):
            module.register_parameter("value", nn.Parameter(4.))
        with self.assertRaises(KeyError):
            module.add_module("value", nn.Module())
        self.assertIs(module.value, original)

    def test_buffer_registration_requires_module_init(self):
        class MissingInit(nn.Module):
            def __init__(self):
                self.register_buffer("value", Tensor(1.))

        with self.assertRaises(AttributeError):
            MissingInit()


if __name__ == "__main__":
    unittest.main()
