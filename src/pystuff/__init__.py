from pystuff.config import Config
from pystuff.logger import Logger
from pystuff.shapehook import ShapeHook
from typing import Literal, Optional


class PyStuff:
    """Unified entry point that initialises Config and Logger together.

    Creates (or reuses) the global :class:`Config` singleton and a new
    :class:`Logger` instance wired to the same project name, so both share
    consistent settings from a single call.

    Attributes:
        config: The global :class:`Config` singleton.
        logger: The :class:`Logger` instance for this project.

    Usage::

        ps = PyStuff(
            project_name="my_exp",
            log_level="info",
            report_to="tb",
        )
        ps.config.parse_cli_args()

        @Config.configurable
        class Trainer:
            def __init__(self, lr=0.001): ...

        ps.logger("Training started")
    """

    def __init__(
        self,
        project_name: str,
        log_level: Literal["debug", "info", "warning", "error"] = "info",
        report_to: Literal["all", "wb", "tb", "none"] | None = None,
        remote_logger_run_name: Optional[str] = None,
        tensorboard_log_dir: str = "./tb",
        log_separator: str = "|",
        log_multi_line: bool = True,
    ):
        """
        Args:
            project_name: Shared project name used for both Logger and W&B/TB runs.
            log_level: Minimum severity level printed to console.
            report_to: Remote backend — ``"wb"``, ``"tb"``, ``"all"``, or ``"none"``.
            remote_logger_run_name: Override the run name for W&B / TensorBoard.
            tensorboard_log_dir: Root directory for TensorBoard event files.
            log_separator: Column separator used in single-line log output.
            log_multi_line: Format dict logs as pretty-printed JSON when ``True``.
        """
        self.config: Config = Config.instance()
        self.logger: Logger = Logger(
            project_name=project_name,
            log_level=log_level,
            report_to=report_to,
            remote_logger_run_name=remote_logger_run_name,
            tensorboard_log_dir=tensorboard_log_dir,
            separator=log_separator,
            multi_line=log_multi_line,
        )
        self.config._log = self.logger

    def parse(self, args=None) -> 'PyStuff':
        """Parse CLI arguments and return self for chaining.

        Args:
            args: Explicit argument list (defaults to ``sys.argv``).
                  Pass ``[]`` in tests.
        """
        self.config.parse_cli_args(args)
        return self


def main() -> None:
    print("Hello from pystuff!")
