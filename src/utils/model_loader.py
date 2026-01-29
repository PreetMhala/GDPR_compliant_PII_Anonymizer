import importlib
import logging
import os
import sys
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

def discover_model_recognizers(supported_languages: List[str]):
    """
    Dynamically discover and load custom model-based recognizers.

    It searches for Python files in the 'models' directory that follow this convention:
    - File starts with an underscore (_)
    - Defines a function named `get_recognizers(supported_languages)`

    Example:
    models/
        _bertic_recognizer.py
        _nerkor_hu.py
        _gr_nlp_recognizer.py

    These files must each define:
        def get_recognizers(supported_languages: List[str]) -> List[Recognizer]:
            ...

    Args:
        supported_languages: List of languages supported in the current run (e.g., ["el", "hr", "hu"])

    Returns:
        A flat list of instantiated recognizer objects
    """
    recognizers = []
    models_path = Path(__file__).parent.parent / "models"

    if not models_path.exists():
        logger.warning("[PRESIDIO] Models folder not found: %s", models_path)
        return recognizers

    # Ensure models path is in sys.path so importlib works
    sys.path.insert(0, str(models_path.resolve()))

    for filename in os.listdir(models_path):
        if filename.startswith("_") and filename.endswith(".py"):
            module_name = filename[:-3]  # Remove .py
            try:
                module = importlib.import_module(module_name)
                if hasattr(module, "get_recognizers"):
                    logger.info("[PRESIDIO] Discovering recognizers from module: %s", module_name)
                    new_recognizers = module.get_recognizers(supported_languages)
                    recognizers.extend(new_recognizers)
                else:
                    logger.debug("[PRESIDIO] Skipping %s (no get_recognizers found)", module_name)
            except Exception as e:
                logger.exception("[PRESIDIO] Error loading recognizer module: %s", module_name)
    return recognizers
