from ruamel.yaml import YAML
from pathlib import Path

yaml = YAML()
yaml.indent(mapping=2, sequence=4, offset=2)
yaml.preserve_quotes = True
yaml.default_flow_style = False


def reformat_yaml_file(file_path: Path):
    with file_path.open("r", encoding="utf-8") as f:
        data = yaml.load(f)

    # Construct new filename with _formatted suffix
    formatted_path = file_path.with_name(file_path.stem + "_formatted.yaml")

    with formatted_path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f)

    print(f"Reformatted file saved as: {formatted_path.name}")


def reformat_all_yaml_in_dir(folder: Path):
    for path in folder.glob("recognizer_en_de_config.yaml"):
        reformat_yaml_file(path)


if __name__ == "__main__":
    yaml_dir = Path("yaml")  # Adjust if needed
    reformat_all_yaml_in_dir(yaml_dir)
