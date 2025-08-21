pip install uv

uv init

uv python install 3.10
uv python pin 3.10
uv venv --python 3.10

uv add -r requirements.txt

uv run streamlit run predict_streamlit.py