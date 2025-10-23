# streamlit_app.py

import io
import requests
import nltk
from nltk.tokenize import word_tokenize
import streamlit as st
import speech_recognition as sr
from PIL import Image
from gtts import gTTS

# LangChain imports (latest version)
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.tools import DuckDuckGoSearchRun

# Download NLTK punkt tokenizer
nltk.download("punkt")

# Hugging Face API key for text summarization & image generation
HUGGINGFACE_API_KEY = "YOUR_HUGGINGFACE_API_KEY"
headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

# ----------------- Helper Functions -----------------

def split_into_meaningful_words(text):
    words = word_tokenize(text)
    meaningful_words = [word for word in words if word.isalnum()]
    return ", ".join(meaningful_words)

def text_summarization_query(payload):
    API_URL = "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-12-6"
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def text_to_image_query(payload):
    API_URL = "https://api-inference.huggingface.co/models/artificialguybr/IconsRedmond-IconsLoraForSDXL"
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content

# ----------------- Streamlit Page Setup -----------------
st.set_page_config(page_title="DiagnoAI", page_icon="🤖", layout="centered")
st.title("🤖 DiagnoAI : Health first!")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Display previous chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ----------------- Audio Upload -----------------
audio_file = st.file_uploader("Upload your audio file", type="wav")
recognizer = sr.Recognizer()

if audio_file:
    st.audio(audio_file)

    # Transcribe audio
    with st.spinner("Transcribing..."):
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
        transcribed_text = recognizer.recognize_google(audio_data)
    st.session_state.messages.append({"role": "user", "content": transcribed_text})
    st.chat_message("user").write(transcribed_text)

    # ----------------- LLM Chat Setup -----------------
    llm = ChatOpenAI(temperature=0.2)  # Latest LangChain ChatOpenAI
    transcription_agent = initialize_agent(
        tools=[DuckDuckGoSearchRun(name="Search")],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=True,
    )

    output_filename = "Output_Audio.wav"

    # Run agent & generate responses
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

        # Get chat response
        with st.spinner("Generating response..."):
            transcription_response = transcription_agent.run(
                st.session_state.messages, callbacks=[st_cb]
            )
            st.session_state.messages.append({"role": "assistant", "content": transcription_response})
        st.write(transcription_response)

        # Text-to-speech
        with st.spinner("Generating voice output..."):
            speech = gTTS(text=transcription_response, lang="en", slow=False)
            speech.save(output_filename)
        st.audio(output_filename)

        # Text-to-image
        with st.spinner("Generating image output..."):
            summarized_text = text_summarization_query({
                "inputs": str(transcription_response) + "-- Please summarize the given text into actionable keywords. Should not exceed 20 words.",
                "options": {"wait_for_model": True},
            })
            # HuggingFace summarization returns a list with dicts
            if isinstance(summarized_text, list) and "summary_text" in summarized_text[0]:
                summarized_text = summarized_text[0]["summary_text"]
            prompt_words = split_into_meaningful_words(str(summarized_text))
            image_bytes = text_to_image_query({
                "inputs": prompt_words + "1 human, english language, exercise, healthy diet, medicines, vegetables, fruits",
                "options": {"wait_for_model": True},
            })
            image_response = Image.open(io.BytesIO(image_bytes))
        st.image(image_response, use_column_width=True)
