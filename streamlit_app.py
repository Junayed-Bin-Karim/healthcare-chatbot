# streamlit_app.py

import io
import requests
import nltk
from nltk.tokenize import word_tokenize
import streamlit as st
import speech_recognition as sr
from PIL import Image
from gtts import gTTS

# Latest LangChain imports
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import DuckDuckGoSearchRun

nltk.download("punkt")

# ---------------- Hugging Face API ----------------
HUGGINGFACE_API_KEY = "YOUR_HUGGINGFACE_API_KEY"
headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

def split_into_meaningful_words(text):
    words = word_tokenize(text)
    return ", ".join([w for w in words if w.isalnum()])

def text_summarization_query(payload):
    API_URL = "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-12-6"
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def text_to_image_query(payload):
    API_URL = "https://api-inference.huggingface.co/models/artificialguybr/IconsRedmond-IconsLoraForSDXL"
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content

# ---------------- Streamlit Setup ----------------
st.set_page_config(page_title="DiagnoAI", page_icon="🤖", layout="centered")
st.title("🤖 DiagnoAI : Health first!")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

audio_file = st.file_uploader("Upload your audio file", type="wav")
recognizer = sr.Recognizer()

if audio_file:
    st.audio(audio_file)

    with st.spinner("Transcribing..."):
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
        transcribed_text = recognizer.recognize_google(audio_data)

    st.session_state.messages.append({"role": "user", "content": transcribed_text})
    st.chat_message("user").write(transcribed_text)

    # ---------------- Chat Agent ----------------
    llm = ChatOpenAI(temperature=0.2)
    transcription_agent = initialize_agent(
        tools=[DuckDuckGoSearchRun(name="Search")],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=False,
    )

    output_filename = "Output_Audio.wav"

    with st.spinner("Generating response..."):
        transcription_response = transcription_agent.run(st.session_state.messages)
        st.session_state.messages.append({"role": "assistant", "content": transcription_response})
        st.chat_message("assistant").write(transcription_response)

    # ---------------- Text-to-Speech ----------------
    with st.spinner("Generating voice output..."):
        speech = gTTS(text=transcription_response, lang="en", slow=False)
        speech.save(output_filename)
    st.audio(output_filename)

    # ---------------- Text-to-Image ----------------
    with st.spinner("Generating image output..."):
        summarized_text = text_summarization_query({
            "inputs": str(transcription_response) + "-- Summarize into actionable keywords under 20 words.",
            "options": {"wait_for_model": True},
        })

        if isinstance(summarized_text, list) and "summary_text" in summarized_text[0]:
            summarized_text = summarized_text[0]["summary_text"]

        prompt_words = split_into_meaningful_words(str(summarized_text))
        image_bytes = text_to_image_query({
            "inputs": prompt_words + "1 human, english language, exercise, healthy diet, medicines, vegetables, fruits",
            "options": {"wait_for_model": True},
        })
        image_response = Image.open(io.BytesIO(image_bytes))

    st.image(image_response, use_column_width=True)
