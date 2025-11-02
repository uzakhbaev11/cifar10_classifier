python -m venv venv
source venv/bin/activate # или venv\Scripts\activate на Windows
pip install -r requirements.txt


# запустить ноутбук (если нужно)
jupyter lab


# или запустить приложение
streamlit run app/streamlit_app.py
