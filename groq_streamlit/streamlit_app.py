import streamlit as st
from typing import Generator
from groq import Groq

# Page setup
st.set_page_config(
    page_title="Groq Chat — Turbo Mode",
    page_icon="🏁",
    layout="wide",
)

# Inject custom dark theme and chat styles
st.markdown("""
    <style>
        /* General styling */
        .main {
            background-color: #0e0e10;
            color: #ffffff;
        }

        .block-container {
            padding: 2rem;
        }

        /* Title styling */
        .title-text {
            font-size: 2.5rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 0.5rem;
            color: #ffffff;
        }

        .subtitle-text {
            text-align: center;
            font-size: 1.2rem;
            color: #aaa;
            margin-bottom: 2rem;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #1f1f24;
            padding: 1rem;
            color: #fff;
        }

        /* Chat messages */
        .chat-bubble-user {
            background-color: #cceeff;
            color: black;
            padding: 0.75rem 1rem;
            border-radius: 10px;
            margin-bottom: 0.5rem;
            max-width: 70%;
        }

        .chat-bubble-assistant {
            background-color: #eeeeee10;
            color: #fff;
            padding: 0.75rem 1rem;
            border-radius: 10px;
            margin-bottom: 0.5rem;
            max-width: 70%;
        }

        /* Chat input */
        textarea {
            background-color: #1f1f24 !important;
            color: white !important;
            border-radius: 10px !important;
            border: 1px solid #f04d4d !important;
        }

        /* Slider customization */
        .stSlider > div[data-baseweb="slider"] {
            color: red;
        }

        .stSelectbox {
            color: black;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<div class='title-text'>🏁 Groq Chat — Turbo Mode 🏎️</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle-text'>Chat with Groq models in real-time!</div>", unsafe_allow_html=True)

# Groq client
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# Chat & model states
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

# Model list
models = {
    "gemma2-9b-it": {"name": "Gemma2-9b-it", "tokens": 8192, "developer": "Google"},
    "llama-3.3-70b-versatile": {"name": "LLaMA3.3-70b-versatile", "tokens": 128000, "developer": "Meta"},
    "llama-3.1-8b-instant": {"name": "LLaMA3.1-8b-instant", "tokens": 128000, "developer": "Meta"},
    "llama3-70b-8192": {"name": "LLaMA3-70b-8192", "tokens": 8192, "developer": "Meta"},
    "llama3-8b-8192": {"name": "LLaMA3-8b-8192", "tokens": 8192, "developer": "Meta"},
}

# Sidebar Settings
with st.sidebar:
    st.header("⚙️ Settings")

    model_option = st.selectbox(
        "Choose a model:",
        options=list(models.keys()),
        format_func=lambda x: models[x]["name"],
        index=3
    )

    max_tokens_range = models[model_option]["tokens"]
    max_tokens = st.slider(
        "🔋 Max Tokens",
        min_value=512,
        max_value=max_tokens_range,
        value=min(4096, max_tokens_range),
        step=512
    )

    st.markdown(
        f"""
        <div style='padding: 10px; background-color: #2e2e35; border-radius: 10px;'>
            <b>🧠 Using {models[model_option]['name']}</b><br>
            <i>by {models[model_option]['developer']}</i>
        </div>
        """,
        unsafe_allow_html=True
    )

# Reset chat if model changes
if st.session_state.selected_model != model_option:
    st.session_state.messages = []
    st.session_state.selected_model = model_option

# Display chat history
for message in st.session_state.messages:
    avatar = '🤖' if message["role"] == "assistant" else '🧑‍💻'
    bubble_class = 'chat-bubble-assistant' if message["role"] == "assistant" else 'chat-bubble-user'
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(f"<div class='{bubble_class}'>{message['content']}</div>", unsafe_allow_html=True)

# Generator for streamed response
def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# User input
if prompt := st.chat_input("Type your message here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar='🧑‍💻'):
        st.markdown(f"<div class='chat-bubble-user'>{prompt}</div>", unsafe_allow_html=True)

    try:
        chat_completion = client.chat.completions.create(
            model=model_option,
            messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
            max_tokens=max_tokens,
            stream=True
        )

        with st.chat_message("assistant", avatar='🤖'):
            chat_responses_generator = generate_chat_responses(chat_completion)
            full_response = st.write_stream(chat_responses_generator)

    except Exception as e:
        st.error(e, icon="🚨")

    if isinstance(full_response, str):
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    else:
        combined_response = "\n".join(str(item) for item in full_response)
        st.session_state.messages.append({"role": "assistant", "content": combined_response})
