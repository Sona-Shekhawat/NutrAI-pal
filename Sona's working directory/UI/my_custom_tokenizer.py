import re
import spacy

nlp = spacy.load('en_core_web_sm')

def custom_tokenizer(text):
    #text cleaning
    text = text.lower()
     # Replace unicode superscript fractions and other weird chars
    text = re.sub(r'[¼½¾⅓⅔⁄¹²³⁴⁵⁶⁷⁸⁹⁰]', ' ', text)

    # Separate numbers stuck to units or words (e.g., "100ml" → "100 ml")
    text = re.sub(r'(\d+)([a-zA-Z]+)', r'\1 \2', text)

    # Remove standalone numbers and units
    text = re.sub(r'\b\d+/?\d*\b', ' ', text)
    text = re.sub(r'\b(?:ml|l|tsp|tbsp|cup|cups|g|kg|oz|gram|grams|pinch|cm|inch|store|style)\b', '', text)

    # Remove hyphens or leading punctuation leftovers
    text = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', text)

    # Remove extra symbols, keep only words
    text = re.sub(r'[^\w\s]', ' ', text)

    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # spacy to remove stop words and anythings except NOUN and PROPN
    # return lemmatized tokens
    text = nlp(text)
    tokens = [token.lemma_ for token in text 
              if not token.is_stop 
              and not token.is_punct 
              and token.pos_ in {'NOUN','PROPN'}]
    
    return tokens 