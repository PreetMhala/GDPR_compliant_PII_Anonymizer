# src/log_utils.py
from src.logger_config import setup_logger
from datetime import datetime
import uuid

logger = setup_logger()


def get_timestamp():
    """Return timestamp like 'May 9, 2025 @ 16:48:20.678'."""
    now = datetime.utcnow()
    return now.strftime("%b %-d, %Y @ %H:%M:%S.") + f"{int(now.microsecond / 1000):03d}"


def log_event(
        step,
        event,
        message,
        natco_code,
        usecase_version,
        log_level="Info",
        application_id="anonymization-service",
        source_intent=None,
        traceId=None,
        model=None,
        k8s_container=None,  # Placeholder – will be added externally by DevOps
):
    """
    Structured logger for all anonymization events.

    Parameters:
    - model (str|None): Optional. Model name (e.g., "bertic", "presidio") to include in logs.
    """
    log_payload = {
        "source_intent": source_intent,  # Optional
        "step": step,  # Required
        "k8s_container": k8s_container,  # Optional/Externally populated
        "application_id": application_id,  # Required
        "traceId": traceId or str(uuid.uuid4()),  # Optional (fallback generated)
        "event": event,  # Required
        "natco_code": natco_code,  # Required
        "usecase_version": usecase_version,  # Required
        "message": message,  # Required
        "timestamp": get_timestamp(),  # Required ISO8601
        "log_level": log_level.capitalize()  # Normalize case
    }

    # Add model only if not None to keep logs clean
    if model is not None:
        log_payload["model"] = model

    # Log as JSON-style dict
    if log_payload["log_level"] == "Error":
        logger.error(log_payload)
    else:
        logger.info(log_payload)
