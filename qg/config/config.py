import pathlib
import logging
from logging import FileHandler
import sys
import torch
PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[2]

import time
from datetime import date
today = str(date.today())
now = time.time()

### LOGGER ###
FORMATTER = logging.Formatter(
    "\n%(asctime)s - %(name)s"
    "\n    %(levelname)s -"
    "\n        function: %(funcName)s@%(lineno)d"
    "\n        %(message)s"
    "\n"
    )
LOG_DIR = PACKAGE_ROOT / 'logs'
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / 'transformer.log'

def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler

def get_file_handler():
    # file_handler = TimedRotatingFileHandler(
    #     LOG_FILE, when='midnight')
    file_handler = FileHandler(LOG_FILE)
    file_handler.setFormatter(FORMATTER)
    file_handler.setLevel(logging.INFO)
    #file_handler.setLevel(logging.WARNING)
    return file_handler

def get_logger(*, logger_name):
    logger = logging.getLogger(logger_name)

    logger.setLevel(logging.INFO)

    logger.addHandler(get_console_handler())
    logger.addHandler(get_file_handler())
    logger.propagate = False

    return logger

_logger = get_logger(logger_name=__name__)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
_logger.info(f"""Day: {today}, time: {now}, running in device: {device}""")


LEARNINGQ_DATA_DIR = PACKAGE_ROOT/"qg"/"LearningQ_data"