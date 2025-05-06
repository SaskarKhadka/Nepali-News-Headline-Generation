from app.config.config import settings
import sentencepiece as spm

def get_tokenizer():
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(settings.TOKENIZER_PATH)
    print("---- Tokenizer Loaded ----")
    return tokenizer

tokenizer = get_tokenizer()