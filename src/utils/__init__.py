SPACY_MODELS = {
    "en": "en_core_web_lg",  # English
    "ca": "ca_core_news_lg",  # Catalan
    "zh": "zh_core_web_lg",   # Chinese
    "hr": "hr_core_news_lg",  # Croatian - natco
    "da": "da_core_news_lg",  # Danish
    "nl": "nl_core_news_lg",  # Dutch
    "fi": "fi_core_news_lg",  # Finnish
    "fr": "fr_core_news_lg",  # French - natco
    "de": "de_core_news_lg",  # German - natco
    "el": "el_core_news_lg",  # Greek - natco
    "it": "it_core_news_lg",  # Italian - natco
    "ja": "ja_core_news_lg",  # Japanese
    "ko": "ko_core_news_lg",  # Korean
    "lt": "lt_core_news_lg",  # Lithuanian
    "mk": "mk_core_news_lg",  # Macedonian - natco
    "nb": "nb_core_news_lg",  # Norwegian Bokmål
    "pl": "pl_core_news_lg",  # Polish - natco
    "pt": "pt_core_news_lg",  # Portuguese
    "ro": "ro_core_news_lg",  # Romanian
    "ru": "ru_core_news_lg",  # Russian
    "sl": "sl_core_news_lg",  # Slovenian
    "es": "es_core_news_lg",  # Spanish - natco
    "sv": "sv_core_news_lg",  # Swedish
    "uk": "uk_core_news_lg", # Ukrainian
    "hu": "hu_core_news_lg",  # Hungarian - natco
}
FLAIR_MODELS = {
    # "en": "beki/flair-pii-distilbert",
    "en": "flair/ner-english",
    "da": "flair/ner-danish", # not required natco
    "nl": "flair/ner-dutch", # not required natco
    "fr": "flair/ner-french",
    "de": "flair/ner-german",
    "es": "flair/ner-spanish",
}

BERTIC_MODELS = {
    "hr": "classla/bcms-bertic-ner"
}

NERKOR_MODELS = {
    "hu": "novakat/nerkor-cars-onpp-hubert"
}

SUPPORTED_LANGUAGES = ["en", "ca", "zh", "hr", "da", "nl", "fi", "fr", "de", "el", "it", "ja", "ko", "lt", "mk", "nb",
                       "pl", "pt", "ro", "ru", "sl", "es", "sv", "uk"]
