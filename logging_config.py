import atexit
import logging
import os
import queue
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler

_LOG_QUEUE = None
_LOG_LISTENER = None
_CONFIGURED = False


def setup_app_logging(log_dir="./pytracker_logs", level=logging.INFO):
    """
    Configure process-wide logging with queue-based handlers.

    Returns
    =======
    logging.Logger
        Base application logger (`pytracker`).
    """
    global _LOG_QUEUE, _LOG_LISTENER, _CONFIGURED

    if _CONFIGURED:
        return logging.getLogger("pytracker")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=False)
        print(f"   ~~~   Created log directory: {os.path.abspath(log_dir)}")
    else:
        print(f"   ~~~   Log directory already exists, appending: {os.path.abspath(log_dir)}")
        # raise FileExistsError(f"Using existing log directory: {os.path.abspath(log_dir)}")
    
    log_queue = queue.Queue(-1)
    queue_handler = QueueHandler(log_queue)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(queue_handler)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        os.path.join(log_dir, "app.log"),
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    listener = QueueListener(log_queue, console_handler, file_handler, respect_handler_level=True)
    listener.start()

    _LOG_QUEUE = log_queue
    _LOG_LISTENER = listener
    _CONFIGURED = True

    atexit.register(shutdown_app_logging)

    app_logger = logging.getLogger("pytracker")
    app_logger.info("Logging initialized. log_dir=%s", os.path.abspath(log_dir))
    return app_logger


def shutdown_app_logging():
    """
    Stop logging listener and flush any remaining log records.
    """
    global _LOG_LISTENER, _CONFIGURED

    if _LOG_LISTENER is not None:
        _LOG_LISTENER.stop()
        _LOG_LISTENER = None

    _CONFIGURED = False
