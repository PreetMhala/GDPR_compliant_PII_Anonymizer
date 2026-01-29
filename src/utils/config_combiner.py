from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedSeq, CommentedMap
from pathlib import Path
from copy import deepcopy

yaml = YAML()
yaml.indent(mapping=2, sequence=4, offset=2)
yaml.preserve_quotes = True
yaml.default_flow_style = False


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.load(f) or {}


def save_yaml(data, path: Path):
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f)
    print(f"Combined config saved to: {path}")


def save_merged_config(merged_data, output_path: Path):
    save_yaml(merged_data, output_path)


def merge_lists_unique(list1, list2):
    return list(dict.fromkeys((list1 or []) + (list2 or [])))


def replace_keywords_recursive(data, section, anchors):
    if isinstance(data, dict):
        if "name" in data and "regex" in data:
            data.pop("keywords", None)
            data.pop("keyword", None)

            alias_key, keyword_field, should_attach = should_attach_keyword_alias(section, data.get("name"), anchors)
            if should_attach:
                anchor_ref = f"shared_keywords_{alias_key}"
                if anchor_ref in anchors:
                    data[keyword_field] = anchors[anchor_ref]
                else:
                    print(f"[Warning] Anchor '{anchor_ref}' not found for section '{section}', entity '{data.get('name')}'")

        for v in data.values():
            replace_keywords_recursive(v, section, anchors)

    elif isinstance(data, list):
        for item in data:
            replace_keywords_recursive(item, section, anchors)


def should_attach_keyword_alias(section, name, anchors):
    alias_map = {
        "passport_list": ("passport_context", "keyword"),
        "phone_recognizer": ("phonenumber", "keywords"),
        "iban_recognizer": ("iban", "keywords"),
        "zip": ("zip_context", "keywords"),
        "id_recognizer": ("id", "keywords"),
        "ian_recognizer": ("ian", "keywords"),
        "imei_recognizer": ("imei", "keywords"),
        "network_address_recognizers": ("network_address", "keywords"),
        "billing_number_recognizer": ("billing_number", "keywords"),
        "credit_card_recognizer": ("credit_card", "keywords"),
        "email_recognizer": ("email", "keywords"),
        "url_recognizer": ("url", "keywords"),
        "date_croatian_recognizer": ("date_croatian", "keywords"),
        "pattern_custom": ("date_croatian", "keywords"),
    }

    if name and "passport" in name.lower():
        return "passport_context", "keyword", True

    if section in alias_map:
        return alias_map[section][0], alias_map[section][1], True

    if section.endswith("_recognizer") or section.endswith("_recognizers"):
        base_name = section.replace("_recognizer", "").replace("_recognizers", "")
        inferred_anchor = f"shared_keywords_{base_name}"
        if inferred_anchor in anchors:
            return base_name, "keywords", True

    return None, None, False


def merge_shared_keywords(common, natco):
    merged = {}
    all_keys = set(common.keys()) | set(natco.keys())
    possible_keys = [k for k in all_keys if k.startswith("shared_keywords_")]

    for key in possible_keys:
        combined = merge_lists_unique(common.get(key, []), natco.get(key, []))
        if combined:
            anchored_list = CommentedSeq(combined)
            anchored_list.yaml_set_anchor(key)
            merged[key] = anchored_list
    return merged


def merge_pattern_custom(common, natco):
    patterns_common = common.get("pattern_custom", {}).get("patterns", [])
    patterns_natco = natco.get("pattern_custom", {}).get("patterns", [])
    return {"patterns": patterns_common + patterns_natco}


def merge_subdict_list_by_key(key, common, natco, anchors):
    common_section = deepcopy(common.get(key, {}))
    natco_section = natco.get(key, {})

    for subkey, val_list in natco_section.items():
        if subkey not in common_section:
            common_section[subkey] = deepcopy(val_list)
        else:
            common_val = common_section[subkey]

            if isinstance(common_val, list) and isinstance(val_list, list):
                existing_names = {v.get("name") for v in common_val if isinstance(v, dict)}
                for entry in val_list:
                    if isinstance(entry, dict) and entry.get("name") not in existing_names:
                        common_val.append(deepcopy(entry))

            elif isinstance(common_val, dict) and isinstance(val_list, dict):
                for inner_key, inner_list in val_list.items():
                    if inner_key not in common_val:
                        common_val[inner_key] = deepcopy(inner_list)
                    else:
                        existing_names = {v.get("name") for v in common_val[inner_key] if isinstance(v, dict)}
                        for entry in inner_list:
                            if isinstance(entry, dict) and entry.get("name") not in existing_names:
                                common_val[inner_key].append(deepcopy(entry))
            else:
                common_section[subkey] = deepcopy(val_list)

    replace_keywords_recursive(common_section, key, anchors)
    return common_section


def merge_config(common_config_path: Path, natco_config_path: Path):
    common = load_yaml(common_config_path)
    natco = load_yaml(natco_config_path)

    merged = CommentedMap()

    merged["SUPPORTED_LANGUAGES"] = merge_lists_unique(
        common.get("SUPPORTED_LANGUAGES", []),
        natco.get("SUPPORTED_LANGUAGES", [])
    )

    merged["PRESIDIO"] = common.get("PRESIDIO", {})
    if "PRESIDIO" in natco:
        merged["PRESIDIO"].update(natco["PRESIDIO"])

    if "BERTIC" in common or "BERTIC" in natco:
        merged["BERTIC"] = common.get("BERTIC") or natco.get("BERTIC")

    merged["ENTITIES_TO_ALLOW"] = merge_lists_unique(
        common.get("ENTITIES_TO_ALLOW", []),
        natco.get("ENTITIES_TO_ALLOW", [])
    )

    shared_keywords = merge_shared_keywords(common, natco)
    anchors = shared_keywords
    for key in sorted(shared_keywords.keys()):
        merged[key] = shared_keywords[key]

    merged["pattern_custom"] = merge_pattern_custom(common, natco)

    title_section = {}
    title_common = common.get("title", {})
    title_natco = natco.get("title", {})
    title_section["titles_list"] = merge_lists_unique(
        title_common.get("titles_list", []),
        title_natco.get("titles_list", [])
    )
    title_section["title_patterns"] = title_common.get("title_patterns", [])
    title_section["title_context"] = title_common.get("title_context", [])
    merged["title"] = title_section

    all_keys = set(common.keys()) | set(natco.keys())
    recognizer_keys = [k for k in sorted(all_keys)
                       if k.endswith("_recognizer") or k in ["zip", "passport_list"]]

    for key in recognizer_keys:
        merged[key] = merge_subdict_list_by_key(key, common, natco, anchors)

    if "numerical_placeholders" in common or "numerical_placeholders" in natco:
        np_combined = {}
        np_combined.update(common.get("numerical_placeholders", {}))
        np_combined.update(natco.get("numerical_placeholders", {}))
        merged["numerical_placeholders"] = np_combined

    remaining_keys = sorted(all_keys - set(merged.keys()))
    for key in remaining_keys:
        if key in natco:
            merged[key] = natco[key]
        else:
            merged[key] = common[key]
    print("Merging completed succesfully")
    return merged


