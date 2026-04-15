"""
Singleton configuration manager combining CLI argument parsing, YAML support,
and automatic ``__init__`` injection via a class decorator.

This module exposes a single ``Config`` class that acts as the central hub
for all runtime configuration in a project.  It is designed to be used in
three steps:

1. **Registration** (import time) — decorate classes with
   ``@Config.configurable``; their ``__init__`` parameters are registered as
   namespaced ``argparse`` flags automatically.
2. **Parsing** (entry point) — call ``Config.instance().parse_cli_args()``
   once after all classes are imported.
3. **Injection** (instantiation) — create decorated classes normally; missing
   arguments are transparently filled from the parsed config.
"""

import yaml
import argparse
import inspect
import functools
from typing import Any, Dict, List, Optional, Type
import os, sys


class Config:
    """Singleton configuration manager with CLI argument parsing and YAML support.

    Combines argument registration, parsing, YAML loading, and a class decorator
    that automatically wires ``__init__`` parameters to CLI flags — all in one place.

    Typical usage::

        @Config.configurable
        class Trainer:
            def __init__(self, lr=0.001, epochs=10):
                self.lr = lr
                self.epochs = epochs

        # In the entry point:
        Config.instance().parse_cli_args()
        trainer = Trainer()  # lr/epochs filled from CLI or defaults
    """

    _instance: Optional['Config'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, log=None):
        if self._initialized:
            return

        self._initialized = True
        self.config: Dict[str, Any] = {}
        self._parser = argparse.ArgumentParser(description="Application Configuration")
        self._parsed = False
        self._registered_prefixes: set = set()
        self._log = log if log is not None else print

    # ------------------------------------------------------------------
    # Singleton access
    # ------------------------------------------------------------------

    @classmethod
    def instance(cls) -> 'Config':
        """Return the global singleton, creating it on first call."""
        return cls()

    @classmethod
    def reset(cls) -> 'Config':
        """Reset and return a fresh singleton. Useful in tests."""
        cls._instance = None
        cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Argument registration
    # ------------------------------------------------------------------

    def add_argument(self, name: str, default: Any = None, arg_type: type = str, help_text: str = ""):
        """Register a single typed CLI argument.

        Args:
            name: Flag name (without ``--`` prefix).
            default: Default value when the flag is not provided.
            arg_type: Type callable used by argparse to coerce the string value.
            help_text: Description shown in ``--help`` output.
        """
        self._parser.add_argument(f"--{name}", type=arg_type, default=default, help=help_text)
        self.config[name] = default

    def add_arguments_from_dict(self, arguments: Dict[str, Dict[str, Any]]):
        """Register multiple arguments from a mapping of name → kwargs.

        Each value is unpacked as keyword arguments to :meth:`add_argument`::

            cfg.add_arguments_from_dict({
                "lr":     {"default": 0.001, "arg_type": float},
                "epochs": {"default": 10,    "arg_type": int},
            })
        """
        for name, params in arguments.items():
            self.add_argument(name, **params)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def parse_cli_args(self, args=None):
        """Parse CLI arguments and store them in :attr:`config`.

        Args:
            args: Explicit list of strings to parse (defaults to ``sys.argv``).
                  Pass an empty list ``[]`` in tests to avoid reading ``sys.argv``.
        """
        parsed = self._parser.parse_args(args)
        self.config.update(vars(parsed))
        self._parsed = True

    # ------------------------------------------------------------------
    # YAML support
    # ------------------------------------------------------------------

    def load_from_yaml(self, yaml_path: str):
        """Merge values from a YAML file into :attr:`config`.

        Missing files produce a warning instead of raising an exception.
        """
        try:
            with open(yaml_path, 'r') as f:
                yaml_data = yaml.safe_load(f) or {}
                self.config.update(yaml_data)
        except FileNotFoundError:
            self._log(f"Warning: Config file not found: {yaml_path}")

    def load_from_multiple_yamls(self, yaml_paths: List[str]):
        """Merge values from multiple YAML files in order."""
        for yaml_path in yaml_paths:
            self.load_from_yaml(yaml_path)

    def save_to_yaml(self, yaml_path: str):
        """Persist the current :attr:`config` to a YAML file."""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

    # ------------------------------------------------------------------
    # Value access and helpers
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Return a config value, falling back to *default* if absent."""
        return self.config.get(key, default)

    def update_config(self, updates: Dict[str, Any]):
        """Merge *updates* directly into :attr:`config`."""
        self.config.update(updates)

    def get_device(self) -> str:
        """Return the best available compute device.

        Reads the ``device`` key (``"auto"``, ``"cuda"``, or ``"mps"``).
        Falls back to ``"cpu"`` when the requested accelerator is unavailable.
        """
        torch = sys.modules.get('torch')
        if torch is None:
            
            return "cpu"

        device = self.config.get("device", "auto")
        if device in ["auto", "cuda"] and torch.cuda.is_available():
            return "cuda"
        if device in ["auto", "mps"] and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def ensure_directory(self, path: str):
        """Create *path* (and any parents) if it does not already exist."""
        if not os.path.exists(path):
            os.makedirs(path)

    def get_checkpoint_path(self) -> str:
        """Return the resolved checkpoint file path, creating its directory.

        Reads ``checkpoint_path``, ``checkpoint_name``, and ``model_name``
        from :attr:`config`.
        """
        checkpoint_path = self.config.get("checkpoint_path", "checkpoints")
        self.ensure_directory(checkpoint_path)
        checkpoint_name = self.config.get("checkpoint_name", None)
        model_name = self.config.get("model_name", "model")
        if checkpoint_name:
            return os.path.join(checkpoint_path, checkpoint_name)
        return f"{os.path.join(checkpoint_path, model_name)}.pth"

    # ------------------------------------------------------------------
    # @Config.configurable decorator
    # ------------------------------------------------------------------

    @staticmethod
    def configurable(cls=None, *, prefix: Optional[str] = None):
        """Class decorator that auto-registers ``__init__`` params as CLI args.

        At **import time** (decoration), each ``__init__`` parameter (except
        ``self``, ``*args``, ``**kwargs``) is registered as a namespaced CLI
        flag on the singleton: ``--<prefix>.<name>``.

        At **instantiation time**, any argument not explicitly supplied by the
        caller is filled from the parsed config.  Explicit arguments always win.

        Args:
            cls: Set automatically when the decorator is used without
                 parentheses (``@Config.configurable``).
            prefix: Namespace for the CLI flags.  Defaults to the lowercase
                    class name (e.g. ``Trainer`` → ``trainer``).

        Raises:
            ValueError: If the resolved prefix is already registered.

        Usage::

            @Config.configurable
            class Trainer:
                def __init__(self, lr=0.001, epochs=10):
                    self.lr, self.epochs = lr, epochs

            @Config.configurable(prefix="opt")
            class Optimizer:
                def __init__(self, lr=0.01):
                    self.lr = lr

            # CLI: python main.py --trainer.lr 0.1 --opt.lr 0.005
        """

        def wrap(klass: Type) -> Type:
            _prefix = prefix if prefix is not None else klass.__name__.lower()
            config = Config.instance()

            if _prefix in config._registered_prefixes:
                raise ValueError(
                    f"Config prefix '{_prefix}' is already registered. "
                    f"Use @Config.configurable(prefix='...') to set a unique prefix."
                )
            config._registered_prefixes.add(_prefix)

            sig = inspect.signature(klass.__init__)
            param_names = []

            for name, param in sig.parameters.items():
                if name == 'self':
                    continue
                if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    continue
                param_names.append(name)
                full_name = f"{_prefix}.{name}"
                default = param.default if param.default is not inspect.Parameter.empty else None
                config._parser.add_argument(f"--{full_name}", default=default)
                config.config[full_name] = default

            original_init = klass.__init__

            @functools.wraps(original_init)
            def new_init(self, *args, **kwargs):
                bound = sig.bind_partial(self, *args, **kwargs)
                for name in param_names:
                    if name not in bound.arguments:
                        value = config.get(f"{_prefix}.{name}")
                        if value is not None:
                            kwargs[name] = value
                original_init(self, *args, **kwargs)

            klass.__init__ = new_init
            klass._config_prefix = _prefix
            return klass

        if cls is not None:
            return wrap(cls)
        return wrap
