# main.py

import streamlit as st
import utils.config as config
from helpers.ui_elements import (
    display_header, display_instructions, display_topics,
    display_warning_if_no_topic_selected
)
from helpers.chat_functions import (
    initialize_openai, initialize_history, send_message,
    display_chat_history
)


# Page configuration in Streamlit
st.set_page_config(
    page_title="Smart Document Search",
    page_icon=":mag_right:",
    layout="centered",
    initial_sidebar_state="expanded"
)


# Load external CSS
with open("helpers/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Display header and instructions
display_header("images/math_green.png")
display_instructions()


# Define and check the selected topic
topic = display_topics()
display_warning_if_no_topic_selected(topic)  # Display warning if no topic is selected


# If a topic is selected, display the rest of the interface
if topic:
    # Initialize OpenAI and chat history
    openai = initialize_openai(config=config)
    initialize_history()

    # Display chat history
    display_chat_history()

    # User message input
    user_input = st.chat_input("Type your question here...")
    if user_input:
        send_message(user_input, openai=st.session_state["openai"], topic=topic)
