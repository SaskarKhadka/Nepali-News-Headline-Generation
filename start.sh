#!/bin/bash

source .venv/bin/activate
export PYTHONPATH=$PWD
uv run -m streamlit run app/Home.py