import unittest
import tempfile
import os
from pystuff.config import Config


class TestConfig(unittest.TestCase):

    def setUp(self):
        self.config = Config()

    def test_add_argument(self):
        self.config.add_argument("test_arg", default=42, arg_type=int, help_text="A test argument")
        self.config.parse_cli_args([])
        self.assertEqual(self.config.get("test_arg"), 42)

    def test_load_from_yaml(self):
        yaml_content = """
        test_key: test_value
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            tmp_path = f.name
        try:
            self.config.load_from_yaml(tmp_path)
            self.assertEqual(self.config.get("test_key"), "test_value")
        finally:
            os.unlink(tmp_path)

    def test_load_from_multiple_yamls(self):
        yaml_content1 = """
        key1: value1
        """
        yaml_content2 = """
        key2: value2
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f1, \
             tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f2:
            f1.write(yaml_content1)
            f2.write(yaml_content2)
            tmp1, tmp2 = f1.name, f2.name
        try:
            self.config.load_from_multiple_yamls([tmp1, tmp2])
            self.assertEqual(self.config.get("key1"), "value1")
            self.assertEqual(self.config.get("key2"), "value2")
        finally:
            os.unlink(tmp1)
            os.unlink(tmp2)

    def test_update_config(self):
        updates = {"new_key": "new_value"}
        self.config.update_config(updates)
        self.assertEqual(self.config.get("new_key"), "new_value")


class TestConfigurable(unittest.TestCase):

    def setUp(self):
        Config.reset()

    def test_defaults_injected(self):
        @Config.configurable
        class MyModel:
            def __init__(self, lr=0.001, epochs=10):
                self.lr = lr
                self.epochs = epochs

        cfg = Config.instance()
        cfg.parse_cli_args([])

        m = MyModel()
        self.assertEqual(m.lr, 0.001)
        self.assertEqual(m.epochs, 10)

    def test_cli_overrides(self):
        @Config.configurable
        class Trainer:
            def __init__(self, lr=0.001, epochs=10):
                self.lr = lr
                self.epochs = epochs

        cfg = Config.instance()
        cfg.parse_cli_args(["--trainer.lr", "0.1", "--trainer.epochs", "50"])

        t = Trainer()
        self.assertEqual(t.lr, "0.1")
        self.assertEqual(t.epochs, "50")

    def test_explicit_args_override_config(self):
        @Config.configurable
        class Trainer:
            def __init__(self, lr=0.001, epochs=10):
                self.lr = lr
                self.epochs = epochs

        cfg = Config.instance()
        cfg.parse_cli_args(["--trainer.lr", "0.5"])

        t = Trainer(lr=0.99)
        self.assertEqual(t.lr, 0.99)

    def test_positional_explicit_arg_not_overridden(self):
        @Config.configurable
        class Trainer:
            def __init__(self, lr=0.001):
                self.lr = lr

        cfg = Config.instance()
        cfg.parse_cli_args(["--trainer.lr", "0.5"])

        t = Trainer(0.99)
        self.assertEqual(t.lr, 0.99)

    def test_custom_prefix(self):
        @Config.configurable(prefix="opt")
        class Optimizer:
            def __init__(self, lr=0.01):
                self.lr = lr

        cfg = Config.instance()
        cfg.parse_cli_args(["--opt.lr", "0.005"])

        o = Optimizer()
        self.assertEqual(o.lr, "0.005")
        self.assertEqual(Optimizer._config_prefix, "opt")

    def test_namespacing_avoids_collisions(self):
        @Config.configurable(prefix="modelA")
        class ModelA:
            def __init__(self, lr=0.001):
                self.lr = lr

        @Config.configurable(prefix="modelB")
        class ModelB:
            def __init__(self, lr=0.01):
                self.lr = lr

        cfg = Config.instance()
        cfg.parse_cli_args(["--modelA.lr", "0.1", "--modelB.lr", "0.2"])

        a = ModelA()
        b = ModelB()
        self.assertEqual(a.lr, "0.1")
        self.assertEqual(b.lr, "0.2")

    def test_duplicate_prefix_raises(self):
        @Config.configurable(prefix="dup")
        class First:
            def __init__(self, x=1):
                self.x = x

        with self.assertRaises(ValueError):
            @Config.configurable(prefix="dup")
            class Second:
                def __init__(self, y=2):
                    self.y = y

    def test_param_without_default(self):
        @Config.configurable
        class Worker:
            def __init__(self, name):
                self.name = name

        cfg = Config.instance()
        cfg.parse_cli_args(["--worker.name", "alice"])

        w = Worker()
        self.assertEqual(w.name, "alice")

    def test_param_without_default_not_provided_raises(self):
        @Config.configurable
        class Worker:
            def __init__(self, name):
                self.name = name

        cfg = Config.instance()
        cfg.parse_cli_args([])

        with self.assertRaises(TypeError):
            Worker()

    def test_kwargs_and_varargs_ignored(self):
        @Config.configurable
        class Flexible:
            def __init__(self, x=1, *args, **kwargs):
                self.x = x

        cfg = Config.instance()
        cfg.parse_cli_args([])

        f = Flexible()
        self.assertEqual(f.x, 1)


if __name__ == "__main__":
    unittest.main()