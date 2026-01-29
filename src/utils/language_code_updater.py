import os
import tempfile
from pathlib import Path
from src.utils.config_combiner import merge_config, save_merged_config


def generate_temp_combined_config(lang_code: str) -> Path:
    """
    Merge the common config and the NATCO-specific config into a temporary file.
    For 'en' and 'de', just return the common config directly.

    Args:
        lang_code (str): Language code like "pl", "hu", "el", etc.

    Returns:
        Path: Path to the temporary YAML config file.
    """
    # Determine source config paths
    base_dir = Path(__file__).resolve().parent.parent.parent / "yaml"
    common_config = base_dir / "recognizer_en_de_config_formatted.yaml"

    # For 'en' and 'de', return the common config without merging
    if lang_code in ["en", "de"]:
        print(f"[✓] Using common config for '{lang_code}': {common_config}")
        return common_config

    # Map of language code to NATCO-specific config
    natco_configs = {
        "pl": base_dir / "recognizer_pl_config_formatted.yaml",
        "hu": base_dir / "recognizer_hu_config_formatted.yaml",
        "el": base_dir / "recognizer_el_config_formatted.yaml",
        "hr": base_dir / "recognizer_hr_config_formatted.yaml",
        "cnr": base_dir / "recognizer_cnr_config_formatted.yaml",
    }
    natco_config = natco_configs.get(lang_code)

    if not natco_config or not natco_config.exists():
        raise ValueError(f"No config found for language: {lang_code}")

    # Create a temporary file for the combined config
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
    temp_path = Path(temp_file.name)
    temp_file.close()  # Close immediately so ruamel.yaml can write to it

    # Merge the configs and write to temp file
    merged_data = merge_config(common_config, natco_config)
    save_merged_config(merged_data, temp_path)

    print(f"[✓] Temporary merged config for '{lang_code}' created at: {temp_path}")
    return temp_path


# Language mapping logic (unchanged)
LANGUAGE_MODEL_MAPPING = {
    "cnr": "hr",  # Use Croatian models for Montenegrin
    # Add more mappings if needed in future
}

def update_lang_code_in_constants(lang_code: str, constants_file: str = None):
    """
    Update the LANG_CODE value in constants.py dynamically.

    Args:
        lang_code (str): Language code like "el", "hr", etc.
        constants_file (str): Path to constants.py file.
    """
    if constants_file is None:
        constants_file = os.path.join(os.path.dirname(__file__), "constants.py")

    # Resolve the actual language model to use
    resolved_lang_code = LANGUAGE_MODEL_MAPPING.get(lang_code, lang_code)

    with open(constants_file, "r") as file:
        lines = file.readlines()

    with open(constants_file, "w") as file:
        for line in lines:
            if line.strip().startswith("LANG_CODE"):
                file.write(f'LANG_CODE = "{resolved_lang_code}"\n')
            else:
                file.write(line)

    print(f"[✓] constants.py LANG_CODE updated to: '{resolved_lang_code}'")
