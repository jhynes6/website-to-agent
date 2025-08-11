import streamlit as st
from src.ui import run_app

# Set page config
st.set_page_config(
    page_title="Laura's AI Chat Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Run the application
if __name__ == "__main__":
    run_app()