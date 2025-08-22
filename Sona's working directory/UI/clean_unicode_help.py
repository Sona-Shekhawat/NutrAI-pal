import unicodedata
import re

def clean_unicode(text):
    # Normalize Unicode to NFKC (breaks ligatures/fractions into components)
    text = unicodedata.normalize("NFKC", text)
    
    # Replace common unicode fractions with ascii equivalents
    replacements = {
        '¼': '1/4',
        '½': '1/2',
        '¾': '3/4',
        '⅓': '1/3',
        '⅔': '2/3',
        '⅛': '1/8',
        '⅜': '3/8',
        '⅝': '5/8',
        '⅞': '7/8',
    }
    for unicode_char, ascii_equiv in replacements.items():
        text = text.replace(unicode_char, ascii_equiv)

    # Remove other non-breaking or invisible unicode
    text = text.replace('\u00A0', ' ')  # non-breaking space
    text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)  # zero-width
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # remove non-ASCII if needed

    return text