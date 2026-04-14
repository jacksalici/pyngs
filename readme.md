# pystuff 🧰

**A collection of utilities for debugging and developing PyTorch models.**

> Not a production library — a personal toolkit that fits my needs. Simpler alternatives exist for each tool, but this keeps everything in one place.

> **torch is optional** — all classes check `sys.modules` at call time rather than at import time, so the package imports cleanly without PyTorch installed.

## Tools 🛠️

- [PyStuff 🚀](#pystuff-) — unified entry point
- [Shape Hooks 🪝](#shape-hooks-)
- [Logger 📜](#logger-)
- [Config ⚙️](#config-️)

---

### PyStuff 🚀

The recommended entry point. Initialises `Config` and `Logger` together under a single project name — no need to wire them separately.

```python
from pystuff import PyStuff

ps = PyStuff(
    project_name="my_exp",
    log_level="info",
    report_to="tb",             # optional: "none" | "wb" | "tb" | "all"
    tensorboard_log_dir="./tb",
)
ps.parse()                      # parses sys.argv into Config

ps.config                       # → Config singleton
ps.logger                       # → Logger instance
```

`ps.parse()` accepts an explicit list for testing: `ps.parse(["--trainer.lr", "0.1"])`.

---

### Shape Hooks 🪝

Attaches forward hooks to every module in a PyTorch model and prints input/output tensor shapes on each forward pass. Hooks can be set to fire **once only** (`one_time=True`), making them useful for a quick shape-trace without cluttering subsequent passes.

```python
from pystuff import ShapeHook
import torch
from torch import nn

model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(32 * 16 * 16, 10),
)

hook_manager = ShapeHook()
hook_manager.register_hooks(model, one_time=True)

output = model(torch.randn(1, 3, 32, 32))
```

```
ShapeHook for Conv2d     in shapes: [[1, 3, 32, 32]]      out shape: [1, 32, 32, 32]
ShapeHook for ReLU       in shapes: [[1, 32, 32, 32]]     out shape: [1, 32, 32, 32]
ShapeHook for MaxPool2d  in shapes: [[1, 32, 32, 32]]     out shape: [1, 32, 16, 16]
ShapeHook for Flatten    in shapes: [[1, 32, 16, 16]]     out shape: [1, 8192]
ShapeHook for Linear     in shapes: [[1, 8192]]           out shape: [1, 10]
```

`ShapeHook` is a singleton — calling `ShapeHook()` anywhere returns the same instance.

---

### Logger 📜

A flexible logging interface that combines Python's `logging` with optional **Weights & Biases** and **TensorBoard** integration. Log strings or metric dicts at different severity levels; only messages at or above the configured level reach the console.

```python
from pystuff import Logger

logger = Logger(
    project_name="my_experiment",
    log_level="info",           # "debug" | "info" | "warning" | "error"
    report_to="tb",             # "none" | "wb" | "tb" | "all"
    tensorboard_log_dir="./tb",
)

logger("Training started", level="info")
logger({"epoch": 1, "loss": 0.42, "acc": 0.91}, level="info")
logger.close()
```

```
2025-04-19 12:52:17 [INFO] | Training started
2025-04-19 12:52:17 [INFO] | epoch: 1 | loss: 0.42 | acc: 0.91
```

| `report_to` | Destination |
|---|---|
| `"none"` | Console only |
| `"wb"` | Console + Weights & Biases |
| `"tb"` | Console + TensorBoard |
| `"all"` | Console + W&B + TensorBoard |

---

### Config ⚙️

A singleton configuration manager that combines **CLI argument parsing** (`argparse`), **YAML loading**, and **automatic `__init__` injection** via the `@Config.configurable` class decorator — all in one `Config` class.

#### How it works

1. **Register** — decorate classes with `@Config.configurable`. Each `__init__` parameter (except `self`, `*args`, `**kwargs`) is automatically registered as a namespaced CLI flag (`--<classname>.<param>`).
2. **Parse** — call `Config.instance().parse_cli_args()` once in your entry point, after all classes are imported.
3. **Inject** — instantiate decorated classes normally. Missing arguments are filled from the parsed config; explicitly passed arguments always win.

```python
# my_classes.py
from pystuff import Config

@Config.configurable
class Trainer:
    def __init__(self, lr=0.001, epochs=10):
        self.lr = lr
        self.epochs = epochs

@Config.configurable(prefix="opt")   # custom prefix to avoid collisions
class Optimizer:
    def __init__(self, lr=0.01):
        self.lr = lr
```

```python
# main.py
from my_classes import Trainer, Optimizer
from pystuff import Config

Config.instance().parse_cli_args()

trainer  = Trainer()           # filled from CLI / defaults
optimizer = Optimizer(lr=1e-3) # explicit arg overrides config
```

```bash
python main.py --trainer.lr 0.1 --trainer.epochs 50 --opt.lr 0.005
```

#### Mixing with YAML

```python
cfg = Config.instance()
cfg.load_from_yaml("config.yaml")   # keys merge into config before parsing
cfg.parse_cli_args()                # CLI overrides YAML values
```

#### Manual argument registration

```python
cfg = Config.instance()
cfg.add_argument("seed", default=42, arg_type=int, help_text="Random seed")
cfg.add_arguments_from_dict({
    "lr":     {"default": 0.001, "arg_type": float},
    "epochs": {"default": 10,    "arg_type": int},
})
cfg.parse_cli_args()
```

#### Helper methods

| Method | Description |
|---|---|
| `cfg.get(key, default)` | Read any config value |
| `cfg.update_config(dict)` | Merge a dict into config |
| `cfg.save_to_yaml(path)` | Persist current config to YAML |
| `cfg.get_device()` | Auto-detect `cuda` / `mps` / `cpu` (returns `"cpu"` silently if torch not imported) |
| `cfg.get_checkpoint_path()` | Resolve checkpoint path, creating dirs |
| `Config.reset()` | Reset singleton (useful in tests) |

---

## Installation 🚀

Not yet on PyPI. Install from source:

```bash
git clone https://github.com/jacksalici/pystuff
cd pystuff
pip install -e .
```

## License

MIT

