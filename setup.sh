#!/bin/bash

curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.10.12
git clone https://github.com/SaskarKhadka/Nepali-News-Headline-Generation.git
cd Nepali-News-Headline-Generation
uv venv
source .venv/bin/activate
export PYTHONPATH=$PWD
uv pip install -r requirements.txt