import emoji

def remove_emojis(text: str) -> str:
    """
    Entfernt Emojis und andere nicht-standard Unicodezeichen aus einem String.
    """
    return emoji.replace_emoji(text, replace='')