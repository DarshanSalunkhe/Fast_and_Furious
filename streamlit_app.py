import streamlit as st

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fast & Furious RAG",
    page_icon="🏎️",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background: #0b0b0f;
        color: white;
    }

    /* Remove top padding */
    .block-container {
        padding-top: 3rem;
        padding-bottom: 2rem;
        max-width: 850px;
    }

    /* Main title */
    .main-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
        color: white;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        font-size: 20px;
        color: #b9b9c3;
        margin-bottom: 10px;
    }

    /* Description */
    .description {
        text-align: center;
        font-size: 16px;
        color: #8f8f9b;
        margin-bottom: 35px;
    }

    /* Logo */
    .logo {
        text-align: center;
        font-size: 65px;
        margin-bottom: 5px;
    }

    /* Chat messages */
    .user-message {
        background: #24242e;
        padding: 12px 16px;
        border-radius: 14px;
        margin: 10px 0;
        color: white;
    }

    .assistant-message {
        background: #17171e;
        padding: 12px 16px;
        border-radius: 14px;
        margin: 10px 0;
        color: white;
        border: 1px solid #292933;
    }

    /* Input box */
    div[data-testid="stChatInput"] {
        border-radius: 15px;
    }

    /* Hide Streamlit branding */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="logo">🏎️</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-title">Fast & Furious</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">RAG Knowledge Database</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="description">
        Ask me anything about the Fast & Furious franchise,
        characters, movies, cars, and events!
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# DISPLAY PREVIOUS MESSAGES
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "E.g., Who is Dominic Toretto?"
)


# ============================================================
# HANDLE USER QUESTION
# ============================================================

if question:

    # Store user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    # Display user message
    with st.chat_message("user"):
        st.markdown(question)

    # Temporary response
    # This will later be replaced with your RAG API call.
    response = (
        "🏎️ Your RAG system is not connected yet. "
        "Once the Fast & Furious RAG backend is deployed, "
        "this response will come directly from your knowledge base."
    )

    # Display assistant response
    with st.chat_message("assistant"):

        with st.spinner("Searching the Fast & Furious knowledge base..."):
            st.markdown(response)

    # Store assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )