import re
import string
import emoji 

def clean_text_for_nlp(text: str) -> str:
    """
    Cleans text for NLP tasks.
    """
    if not text or not isinstance(text, str):
        return ""
    
    cleaned_text = text.lower()
    cleaned_text = re.sub(r'http\S+|www\S+|https\S+', '', cleaned_text, flags=re.MULTILINE)
    cleaned_text = re.sub(r'@\w+', '', cleaned_text)
    cleaned_text = cleaned_text.replace('#', '')

    # --- Remove emojis ---
    cleaned_text = emoji.replace_emoji(cleaned_text, replace=' ') 

    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    cleaned_text = cleaned_text.translate(translator)

    cleaned_text = re.sub(r'\s+', ' ', cleaned_text) 
    cleaned_text = cleaned_text.strip()
    return cleaned_text
