# Pyngs 🧰
> Python Things for easy Python project management.

**A collection of utilities for managing the configuration, logging, and debugging of Python projects.**

- [Config 🎛️](#config-️): A singleton configuration manager combining CLI parsing, YAML loading, and attribute-style access. Optional `@Config.configurable` decorator for auto-registering class `__init__` parameters as CLI flags. 
- [Logger 📠](#logger-): A singleton logging interface that wraps Python's `logging` with optional Weights & Biases and TensorBoard integration.
- [Shape Hooks 🪝](#shape-hooks-) (🚧 experimental): Attach forward hooks to PyTorch modules to print input/output tensor shapes.

---
## Tools

### Config 🎛️

A **singleton** configuration manager combining **CLI argument parsing** (`argparse`), **YAML loading**, **attribute-style access**, and an optional `@Config.configurable` class decorator.

```python
from pyngs import Config

cfg = Config()
cfg.add_argument("lr", default=0.001, arg_type=float)
cfg.add_argument("epochs", default=10, arg_type=int)
cfg.parse_cli_args()

# Dot access, bracket access, or .get()
print(cfg.lr)          # 0.001
print(cfg["epochs"])   # 10
print(cfg.get("missing", 42))  # 42

cfg["new_key"] = "hello"  # bracket assignment
```

#### `@Config.configurable` decorator

Registers `__init__` parameters as `--<classname>.<param>` CLI flags. At instantiation, missing arguments are filled from the parsed config.

```python
@Config.configurable
class Trainer:
    def __init__(self, lr=0.001, epochs=10):
        self.lr, self.epochs = lr, epochs

cfg = Config()
cfg.parse_cli_args()
trainer = Trainer()  # filled from CLI / defaults
```

```bash
python main.py --trainer.lr 0.1 --trainer.epochs 50
```

#### YAML support

```python
cfg.load_from_yaml("config.yaml")  # merge before parsing
cfg.parse_cli_args()               # CLI overrides YAML
cfg.save_to_yaml("snapshot.yaml")
```

#### Methods

| Method | Description |
|---|---|
| `cfg.get(key, default)` | Read with fallback |
| `cfg.update(dict)` | Merge a dict into config |
| `cfg.save_to_yaml(path)` | Persist config to YAML |
| `cfg.load_from_yaml(path)` | Load from YAML |
| `Config.reset()` | Reset singleton (tests) |

### Logger 📠

A **singleton** logging interface that combines Python's `logging` with optional **Weights & Biases** and **TensorBoard** integration. Log strings or metric dicts at different severity levels; only messages at or above the configured level reach the console. Calling `Logger()` multiple times returns the same instance.

```python
from pyngs import Logger

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

### Shape Hooks 🪝

Attaches forward hooks to every module in a PyTorch model and prints input/output tensor shapes on each forward pass. Hooks can be set to fire **once only** (`one_time=True`), making them useful for a quick shape-trace without cluttering subsequent passes.

```python
from pyngs import ShapeHook
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

## Installation 🚀

### From source

```bash
git clone https://github.com/jacksalici/pyngs
cd pyngs
pip install -e .
```

### From PyPI

```bash
pip install pyngs
```

## License

MIT

