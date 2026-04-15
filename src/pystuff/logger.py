"""
Unified logging utility combining console output with optional remote tracking

This module provides a flexible logging interface that combines Python's
standard logging with optional Weights & Biases and TensorBoard integration.
It enables logging text messages at different severity levels and structured
data like metrics and configurations in both local and remote contexts.
"""

import json
import logging
import sys
from typing import Any, Dict, Literal


class Logger:
    """
    Singleton that provides console logging with optional Weights & Biases and TensorBoard integration.
    """

    _instance = None
    
    _logging_levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }

    _wandb = None
    _tb_writer = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        project_name: str,
        log_level: Literal["debug", "info", "warning", "error"] = "info",
        report_to: Literal["all","wb", "tb", "none"] | None = None,
        remote_logger_run_name: str | None = None,
        tensorboard_log_dir: str = "./tb",
        separator: str = "|",
        multi_line: bool = True,
    ):
        """
        Initializes the Logger instance.

        Args:
            project_name: Name of the project for logging.
            log_level: Logging level. Defaults to "info".
            report_to: Remote logger to use. Options: "all" (both Weights & Biases and TensorBoard), "wb" (Weights & Biases), "tb" (TensorBoard), or "none" (no remote logging). Defaults to None.
            remote_logger_run_name: Name for the remote logger run. Defaults to None.
            tensorboard_log_dir: Directory for TensorBoard logs. Defaults to "./runs".
            separator: Separator used in log dict messages. Defaults to "|".
            multi_line: If True, formats log dict messages in multiple lines. Defaults to True.
        """
        if self._initialized:
            return

        self._initialized = True
        self.logger = logging.getLogger(project_name)
        self.logger.setLevel(self._logging_levels.get(log_level, logging.INFO))
        self.separator = separator
        self.multi_line = multi_line
        self._step = 0  # Track steps for TensorBoard

        if not self.logger.handlers:
            #handler = logging.StreamHandler()
            handler = logging.NullHandler()
            formatter = logging.Formatter(
                f"%(asctime)s [%(levelname)s] {separator} %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        if report_to == "wb":
            self._init_wandb(project_name, remote_logger_run_name)
        elif report_to == "tb":
            self._init_tensorboard(tensorboard_log_dir, remote_logger_run_name or project_name)
        elif report_to == "all":
            self._init_wandb(project_name, remote_logger_run_name)
            self._init_tensorboard(tensorboard_log_dir, remote_logger_run_name or project_name)
        elif report_to is None or report_to == "none":
            self.logger.info("No remote logger selected; using console logging only.")
        else:
            self.logger.warning(
                f"Unknown logger preference '{report_to}'. "
                "Falling back to console logging only."
            )

    def print_config(self, config_dict: Dict[str, Any]) -> None:
        """
        Prints the configuration dict to the console and logs it where needed.

        Args:
            config_dict (dict): Configuration dictionary to print.
        """

        if self._wandb:
            self._wandb.config.update(config_dict)

        if self._tb_writer:
            # Log config as text to TensorBoard
            config_text = json.dumps(config_dict, indent=4)
            self._tb_writer.add_text("config", config_text, 0)
            print(config_dict)
            _torch = sys.modules.get('torch')
            c = {}
            _tensor_type = (_torch.Tensor,) if _torch else ()
            for key, value in config_dict.items():
                if isinstance(value, (str, int, float, bool, *_tensor_type)):
                    c[key] = value
                else:
                    c[key] = str(value)
            self._tb_writer.add_hparams(c, {})

        self.logger.info(f"Configuration: {json.dumps(config_dict, indent=4)}")

    def __call__(
        self,
        info: dict | str,   
        level: Literal["debug", "info", "warning", "error"] = "info",
        step: int | None = None,
    ) -> None:
        """
        Send information to the logger.

        This can be used to log metrics, parameters, or any other information.
        It will log to the console, wandb, and TensorBoard (if enabled).

        Args:
            info: Information to log, typically a dictionary of metrics or
                parameters or a string message.
            level: The logging level of the message. Defaults to "info".
                - Available levels: "debug", "info", "warning", "error".
            step: The step number for TensorBoard logging. If None, uses internal counter.
        """
        assert isinstance(info, (dict, str)), "Info must be a dictionary or a string."

        # Use provided step or increment internal counter
        current_step = step if step is not None else self._step
        if step is None:
            self._step += 1

        # Case 1: info is a dictionary
        if isinstance(info, dict):
            if self._wandb:
                self._wandb.log(info)

            if self._tb_writer:
                # Log each metric to TensorBoard
                for key, value in info.items():
                    if isinstance(value, (int, float)):
                        self._tb_writer.add_scalar(key, value, current_step)
                    elif isinstance(value, str):
                        self._tb_writer.add_text(key, value, current_step)

            log_message = (
                f" {self.separator} ".join(
                    [f"{key}: {value}" for key, value in info.items()]
                )
                if not self.multi_line
                else json.dumps(info, indent=4)
            )

        # Case 2: info is a string
        else:
            log_message = info
            if self._tb_writer:
                self._tb_writer.add_text("logs", log_message, current_step)

        logging_level = self._logging_levels.get(level, logging.INFO)

        self.logger.log(logging_level, log_message)

    def close(self):
        """
        Closes all logging resources.
        """
        if self._wandb:
            self._wandb.finish()
        
        if self._tb_writer:
            self._tb_writer.close()

    # ------------------------------------------------------------------
    # Singleton access
    # ------------------------------------------------------------------

    @classmethod
    def instance(cls) -> 'Logger':
        """Return the global singleton, creating it on first call."""
        return cls()

    @classmethod
    def reset(cls) -> 'Logger':
        """Reset and return a fresh singleton. Useful in tests."""
        cls._instance = None
        cls._instance = cls()
        return cls._instance

    def _init_wandb(self, project_name: str, remote_logger_run_name: str | None = None):
        try:
            import wandb

            wandb.init(project=project_name, name=remote_logger_run_name)
            self._wandb = wandb
        except ImportError:
            self.logger.warning(
                "Weights & Biases (wandb) is not installed. "
                "Please install it to use wandb logging."
            )
            self._wandb = None

    def _init_tensorboard(self, log_dir: str, run_name: str):
        try:
            from torch.utils.tensorboard import SummaryWriter
            
            self._tb_writer = SummaryWriter(log_dir=f"{log_dir}/{run_name}")
            self.logger.info(f"TensorBoard logging enabled. Log directory: {log_dir}/{run_name}")
            self.logger.info(
                "To view TensorBoard logs, run:\n"
                f"tensorboard --logdir={log_dir}"
            )
        except ImportError:
            self.logger.warning(
                "TensorBoard (torch.utils.tensorboard) is not installed. "
                "Please install PyTorch to use TensorBoard logging."
            )
            self._tb_writer = None


def demo():
    """
    Demonstrates the usage of the Logger class.
    """
    logger = Logger(
        project_name="LoggerDemo",
        log_level="info",
        logger_preference="tb",
        tensorboard_log_dir="./demo_logs"
    )

    # log string messages
    logger("This is a test log message.", level="info")
    logger("This is a warning message.", level="warning")
    logger("This is an error message.", level="error")

    # log a debug message
    # - this should not appear since log_level is set to "info"
    logger("This is a debug message.", level="debug")

    # log dummy metrics over multiple steps
    for i in range(5):
        metrics = {
            "val_accuracy": 0.85 + i * 0.02,
            "val_loss": 0.15 - i * 0.02,
            "epoch": i,
        }
        logger(metrics, level="info", step=i)

    # log dummy configuration dictionary
    config = {
        "lr": 0.0005,
        "batch_size": 64,
        "num_epochs": 8,
    }
    logger.print_config(config)

    # Close logger resources
    logger.close()
    
    print("\nTo view TensorBoard logs, run:")
    print("tensorboard --logdir=./demo_logs")


if __name__ == "__main__":
    demo()