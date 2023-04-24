import logging
import os
import socket
import sys
import warnings
from datetime import datetime
from typing import Any, Dict, Optional, TypeVar, Union

import rich
import torch
from rich.console import Console, ConsoleRenderable
from rich.highlighter import NullHighlighter
from rich.text import Text
from rich.traceback import Traceback

from .exceptions import OlmoCliError, OlmoError

_log_extra_fields: Dict[str, Any] = {}


def log_extra_field(field_name: str, field_value: Any) -> None:
    global _log_extra_fields
    if field_value is None:
        if field_name in _log_extra_fields:
            del _log_extra_fields[field_name]
    else:
        _log_extra_fields[field_name] = field_value


def setup_logging() -> None:
    log_extra_field("hostname", socket.gethostname())
    old_log_record_factory = logging.getLogRecordFactory()

    def log_record_factory(*args, **kwargs) -> logging.LogRecord:
        record = old_log_record_factory(*args, **kwargs)
        for field_name, field_value in _log_extra_fields.items():
            setattr(record, field_name, field_value)
        return record

    logging.setLogRecordFactory(log_record_factory)

    handler: logging.Handler
    if (
        os.environ.get("OLMo_NONINTERACTIVE", False)
        or os.environ.get("DEBIAN_FRONTEND", None) == "noninteractive"
        or not sys.stdout.isatty()
    ):
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s\t%(hostname)s:%(local_rank)s\t%(name)s:%(lineno)s\t%(levelname)s\t%(message)s"
        )
        formatter.default_time_format = "%Y-%m-%d %H:%M:%S"
        formatter.default_msec_format = "%s.%03d"
        handler.setFormatter(formatter)
    else:
        handler = RichHandler()

    logging.basicConfig(handlers=[handler], level=logging.INFO)

    logzio_token = os.environ.get("LOGZIO_TOKEN", None)
    if logzio_token is not None:
        from logzio.handler import LogzioHandler

        logging.getLogger().addHandler(LogzioHandler(logzio_token))

    logging.captureWarnings(True)


def excepthook(exctype, value, traceback):
    """
    Used to patch `sys.excepthook` in order to log exceptions.
    """
    if issubclass(exctype, KeyboardInterrupt):
        sys.__excepthook__(exctype, value, traceback)
    elif issubclass(exctype, OlmoCliError):
        rich.get_console().print(f"[yellow]{value}[/]", highlight=False)
    elif issubclass(exctype, OlmoError):
        rich.get_console().print(Text(f"{exctype.__name__}:", style="red"), value, highlight=False)
    else:
        logging.getLogger().critical(
            "Uncaught %s: %s", exctype.__name__, value, exc_info=(exctype, value, traceback)
        )


def install_excepthook():
    sys.excepthook = excepthook


def filter_warnings():
    # Torchvision warnings. We don't actually use torchvision.
    warnings.filterwarnings(
        action="ignore",
        message="failed to load.*",
        module="torchvision.io.image",
    )


def set_env_variables():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def prepare_cli_environment():
    rich.reconfigure(width=max(rich.get_console().width, 180), soft_wrap=True)
    setup_logging()
    install_excepthook()
    filter_warnings()
    set_env_variables()


def clean_opt(arg: str) -> str:
    if "=" not in arg:
        arg = f"{arg}=True"
    name, val = arg.split("=", 1)
    name = name.strip("-").replace("-", "_")
    return f"{name}={val}"


class RichHandler(logging.Handler):
    """
    A simplified version of rich.logging.RichHandler from
    https://github.com/Textualize/rich/blob/master/rich/logging.py
    """

    def __init__(
        self,
        *,
        level: Union[int, str] = logging.NOTSET,
        console: Optional[Console] = None,
        markup: bool = False,
    ) -> None:
        super().__init__(level=level)
        self.console = console or rich.get_console()
        self.highlighter = NullHighlighter()
        self.markup = markup

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if hasattr(record.msg, "__rich__") or hasattr(record.msg, "__rich_console__"):
                self.console.print(record.msg)
            else:
                msg: Any = record.msg
                if isinstance(record.msg, str):
                    msg = self.render_message(record=record, message=record.getMessage())
                renderables = [
                    self.get_time_text(record),
                    self.get_level_text(record),
                    self.get_location_text(record),
                    msg,
                ]
                if record.exc_info is not None:
                    tb = Traceback.from_exception(*record.exc_info)  # type: ignore
                    renderables.append(tb)
                self.console.print(*renderables)
        except Exception:
            self.handleError(record)

    def render_message(self, *, record: logging.LogRecord, message: str) -> ConsoleRenderable:
        use_markup = getattr(record, "markup", self.markup)
        message_text = Text.from_markup(message) if use_markup else Text(message)

        highlighter = getattr(record, "highlighter", self.highlighter)
        if highlighter:
            message_text = highlighter(message_text)

        return message_text

    def get_time_text(self, record: logging.LogRecord) -> Text:
        log_time = datetime.fromtimestamp(record.created)
        time_str = log_time.strftime("[%Y-%m-%d %X]")
        return Text(time_str, style="log.time", end=" ")

    def get_level_text(self, record: logging.LogRecord) -> Text:
        level_name = record.levelname
        level_text = Text.styled(level_name.ljust(8), f"logging.level.{level_name.lower()}")
        level_text.style = "log.level"
        level_text.end = " "
        return level_text

    def get_location_text(self, record: logging.LogRecord) -> Text:
        name_and_line = f"{record.name}:{record.lineno}" if record.name != "root" else "root"
        text = f"[{name_and_line}, rank={record.local_rank}]"  # type: ignore
        return Text(text, style="log.path")


def seed_all(seed: int):
    """Seed all rng objects."""
    import random

    import numpy as np

    if seed < 0 or seed > 2**32 - 1:
        raise ValueError(f"Seed {seed} is invalid. It must be on [0; 2^32 - 1]")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.manual_seed may call manual_seed_all but calling it again here
    # to make sure it gets called at least once
    torch.cuda.manual_seed_all(seed)


T = TypeVar("T")


def move_to_device(o: T, device: torch.device) -> T:
    if isinstance(o, torch.Tensor):
        return o.to(device)  # type: ignore[return-value]
    elif isinstance(o, dict):
        return {k: move_to_device(v, device) for k, v in o.items()}  # type: ignore[return-value]
    elif isinstance(o, list):
        return [move_to_device(x, device) for x in o]  # type: ignore[return-value]
    elif isinstance(o, tuple):
        return tuple((move_to_device(x, device) for x in o))  # type: ignore[return-value]
    else:
        return o
