import streamlit as st
import base64


def get_base64_image(image_path):
    """
    Converts an image to a base64-encoded string.
    """
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("utf-8")


def display_header(image_path):
    """
    Displays a header with title, subtitle, and an image in base64 format.
    """
    image_base64 = get_base64_image(image_path)
    st.markdown(f"""
        <div class="title-container">
            <h1 class="title">Smart Document Search</h1>
            <h3 class="subtitle">Your semantic search tool for corporate documents</h3>
            <img class="logo-image" src="data:image/png;base64,{image_base64}" />
            <p class="description">Welcome to the platform for smart document search. Input a question, and the system will search for relevant answers from the available corporate documents.</p>
        </div>
    """, unsafe_allow_html=True)


def display_instructions():
    """
    Displays instructions for effective question formulation in the sidebar.
    """
    with st.sidebar:
        st.header("Guidelines for Effective Questions")
        st.write("""
            - **Be specific**: The more detailed your question, the better the answer.
            - **Use keywords**: Include relevant terms, like project names or technologies, to guide the search.
            - **Provide context**: Add necessary context to make the answer more relevant.
            - **Avoid ambiguity**: Be clear and direct, avoiding multiple interpretations.
        """)


def display_topics():
    """
    Displays a selectbox for topic selection in the sidebar and a footer.
    """
    available_topics = ["", "WINGS", "AGENDA", "TAX", "SALES", "VECTOR_LOADED", "FULL", "BOOKS"]
    topic = st.sidebar.selectbox(
        label="TOPIC",
        options=available_topics,
        format_func=lambda x: "Select a topic" if x == "" else x,
        key="SELECTED_TOPIC"
    )

    # Sidebar footer
    st.sidebar.markdown("<div class='footer'>Powered by MATH TECH</div>", unsafe_allow_html=True)
    
    return topic.lower()


def display_warning_if_no_topic_selected(topic):
    """
    Displays a warning message if no topic is selected.
    """
    if not topic:
        st.markdown(
            """
            <div style='
                background-color: #024914; 
                padding: 10px; 
                border-radius: 5px;
                color: white; 
                font-weight: normal;
                text-align: left;
                font-family: inherit;
                font-size: inherit;
            '>
                Please select a topic to continue.
            </div>
            """, 
            unsafe_allow_html=True
        )

#024914