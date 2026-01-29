import logging
import os
from pythonjsonlogger import jsonlogger

def setup_logger():
    logger = logging.getLogger("anonymization_logger")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        log_file_path = os.path.join(os.path.dirname(__file__), "..", "test/anonymization_log_test.json")
        file_handler = logging.FileHandler(log_file_path)

        # Updated format string — includes all desired fields including k8s_container
        format_fields = [
            "timestamp", "log_level", "application_id", "step", "event",
            "message", "natco_code", "usecase_version", "source_intent",
            "traceId", "model", "k8s_container"
        ]

        formatter = jsonlogger.JsonFormatter(fmt=" ".join(f"%({field})s" for field in format_fields))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
