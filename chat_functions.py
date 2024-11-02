import streamlit as st
from utils.openai_tools import OpenAIGPT
from utils.pdf_tools import PDFReader


def initialize_openai(config):
    """
    Initializes an OpenAIGPT instance and stores it in the Streamlit session state.
    """
    if "openai" not in st.session_state:
        st.session_state["openai"] = OpenAIGPT(
            client_id=config.client_id,
            client_secret=config.client_secret,
            scope=config.scope,
            model_txt=config.model_txt,
            model_embedding=config.model_embedding,
            token_endpoint=config.token_endpoint,
            proxy=config.http_proxy
        )
    return st.session_state["openai"]


def initialize_history():
    """
    Initializes chat history in the Streamlit session state if it doesn't already exist.
    """
    if "history" not in st.session_state:
        st.session_state["history"] = []


def send_message(user_message, openai, topic):
    """
    Processes a user message, handles topic updates, loads documents, initializes FAISS
    database, and displays the assistant's response.
    """
    if "current_topic" not in st.session_state or st.session_state["current_topic"] != topic:
        # Update topic and clear documents and database for reloading
        st.session_state["current_topic"] = topic
        st.session_state.pop("all_documents", None)
        st.session_state.pop("faiss_db", None)

    # Load documents into memory if not already loaded
    if 'all_documents' not in st.session_state and topic != 'vector_loaded':
        with st.spinner('Processing Documents, wait a few seconds...'):
            pdf_reader = PDFReader()
            st.session_state['all_documents'] = pdf_reader.concatenate_documents(
                directory=f"documents/{topic}"
            )

    # Initialize FAISS database in session state if needed
    if "faiss_db" not in st.session_state:
        with st.spinner('Analyzing documents ...'):
            if topic != "vector_loaded":
                st.session_state["faiss_db"] = openai.create_vector_from_documents(
                    st.session_state['all_documents']
                )
            else:
                st.session_state["faiss_db"] = openai.create_vector_from_disk()

    if user_message:
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(user_message)

        placeholder_response = st.empty()
        with placeholder_response.container():
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown("Writing ...")
                response = openai.get_response_from_documents(question=user_message)

        with placeholder_response.container():
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(response)

        st.session_state["history"].append(("You", user_message))
        st.session_state["history"].append(("AI", response))


def display_chat_history():
    """
    Displays the chat history from the Streamlit session state.
    """
    for user, message in st.session_state["history"]:
        with st.chat_message("user" if user == "You" else "assistant", avatar="🧑‍💻" if user == "You" else "🤖"):
            st.markdown(message)
