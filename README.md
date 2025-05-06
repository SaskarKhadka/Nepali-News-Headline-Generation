# Nepali-News-Headline-Generation
## Abstractive Summarization of Nepali News Artciles

In this project I experiment with Attentive Seq2Seq and Transformer models to perform abstractive summarization of Nepali news articles

## Setup Instructions

- Method 1
    - Run the setup.sh script
        - `setup.sh`

    - Inside the project folder
        - Create models and tokenizer folder
            - `mkdir models`
            - `mkdir tokenizer`

        - Add weights of seq2seq and transformer models and the tokenizer to their respective folders
            - Name of seq2seq model = seq2seq.weights.h5
            - Name of Transformer model = transformer.weights.h5
            - Tokenizer must have two files, summarization_50000.model, summarization_50000.vocab
    - Run start.sh
        - `./start.sh`

- Method 2
    1. Install uv, a python package and project manager
        - `curl -LsSf https://astral.sh/uv/install.sh | sh`

    2. Install python 3.13.2
        - `uv python install 3.10.12`

    3. Clone the repository
        - `git clone https://github.com/SaskarKhadka/Nepali-News-Headline-Generation.git`
        - `cd Nepali-News-Headline-Generation`

    2. Create a Virtual Environment
        - `uv venv`

    3. Activate the Virtual Environment
        - `source .venv/bin/activate` (Mac and Linux)
        - `.venv/Scripts/activate` (Windows)

    4. Install Dependencies
        - `uv pip install -r requirements.txt`

    5. Create models and tokenizer folder
        - `mkdir models`
        - `mkdir tokenizer`

    6. Add weights of seq2seq and transformer models to their respective folders
        - Name of seq2seq model = seq2seq.weights.h5
        - Name of Transformer model = transformer.weights.h5
        - Tokenizer must have two files, summarization_50000.model, summarization_50000.vocab

    7. Set Python path to the present working directory
        - `export PYTHONPATH=$PWD`

    8. Run Streamlit App
        - `uv run streamlit run app/Home.py`