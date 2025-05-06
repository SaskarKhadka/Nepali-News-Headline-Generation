import os

class Settings:
    APP_LEVEL_PATH: str = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
    ROOT_LEVEL_PATH: str = os.path.split(APP_LEVEL_PATH)[0]

    TOKENIZER_PATH = os.path.join(ROOT_LEVEL_PATH, 'tokenizer/summarization_50000.model')
    TRANSFORMER_MODEL_PATH = os.path.join(ROOT_LEVEL_PATH, 'models/transformer.weights.h5')
    SEQ2SEQ_MODEL_PATH = os.path.join(ROOT_LEVEL_PATH, 'models/seq2seq.weights.h5')


settings = Settings()