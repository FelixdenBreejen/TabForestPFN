from __future__ import annotations
from pathlib import Path
import sys
import logging
    

def get_logger(log_path: Path) -> logging.Logger:

    logging.setLogRecordFactory(CustomLogRecord)
    logger = logging.getLogger(str(log_path))  # ensures that the logger is unique
    
    if logger.parent.hasHandlers():
        # In a multiprocessing process, the parent logger prints to stdout by default????
        logger.parent.handlers.clear()

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s :: %(levelname)-8s :: %(funcNameMaxWidth)-15s ::   %(message)s')
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


class CustomLogRecord(logging.LogRecord):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.funcNameMaxWidth = self.funcName[:12] + '...' if len(self.funcName) > 15 else self.funcName

    