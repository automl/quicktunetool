import logging
from typing import Optional


def verbosity2loglevel(verbosity: int):
    """Translates verbosity to logging level. Suppresses warnings if verbosity = 0.

    Args:
        verbosity (int): Verbosity level

    Returns:
        int: Logging level
    """
    if verbosity <= 0:  # only errors
        # print("Caution: all warnings suppressed")
        log_level = 40
    elif verbosity == 1:  # only warnings and critical print statements
        log_level = 25
    elif verbosity == 2:  # key print statements which should be shown by default
        log_level = 20
    elif verbosity == 3:  # more-detailed printing
        log_level = 15
    else:
        log_level = 10  # print everything (ie. debug mode)
    return log_level


def set_logger_verbosity(verbosity: int, logger=None):
    """
    Set the verbosity of the logger. If no logger is provided, the root logger is used.

    Args:
        verbosity (int): Verbosity level
        logger (logging.Logger, optional): Logger to set verbosity for. Defaults to None.
    """
    if logger is None:
        logger = logging.root
    if verbosity < 0:
        verbosity = 0
    elif verbosity > 4:
        verbosity = 4
    logger.setLevel(verbosity2loglevel(verbosity))


def add_log_to_file(
    file_path: str,
    logger: Optional[logging.Logger] = None,
    fmt: Optional[str] = None,
    datefmt: Optional[str] = None,
):
    """
    Add a FileHandler to the logger to log to a file in addition to the console.
    If no format is provided, the format is set to: asctime - name: levelname message

    Args:
        file_path (str): Path to the log file
        logger (logging.Logger, optional): Logger to add the file handler to. Defaults to None.
        fmt (str, optional): Format string. Defaults to None.
        datefmt (str, optional): Date format string. Defaults to None.
    """
    if logger is None:
        logger = logging.root
    fh = logging.FileHandler(file_path)
    if fmt is None:
        fmt = "%(asctime)s - %(name)16s: [%(levelname)s] %(message)s"
    if datefmt is None:
        datefmt = "%y.%m.%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def setup_default_logging(
    default_level=logging.INFO,
    fmt: Optional[str] = None,
    datefmt: Optional[str] = None,
):
    """Set up the default logging level and formatter for the root logger.
    If no format is provided, only the message is printed.

    Args:
        default_level (int, optional): Default logging level. Defaults to logging.INFO.
        fmt (str, optional): Format string. Defaults to None.
        datefmt (str, optional): Date format string. Defaults to None.
    """
    if fmt is None:
        fmt = "%(message)s"
        # fmt = "%(asctime)s - %(name)s: [%(levelname)s] %(message)s"
    if datefmt is None:
        datefmt = "%y.%m.%d %H:%M:%S"
    logging.basicConfig(format=fmt, datefmt=datefmt, level=default_level)
