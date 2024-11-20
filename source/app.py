import os
import time
import pygame
from gtts import gTTS
import streamlit as st
import speech_recognition as sr
from googletrans import LANGUAGES, Translator
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# SUMMARIZATION
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def summarize_text(text):
    inputs = tokenizer(text, max_length=1024, return_tensors="pt")
    summary = model.generate(inputs["input_ids"], num_beams=4, max_length=128)
    summary_text = tokenizer.decode(summary[0], skip_special_tokens=True)
    return summary_text


# Streamlit state to accumulate all spoken text across runs
if "total_text" not in st.session_state:
    st.session_state.total_text = ""  # Initialize total text

# TRANSLATION
isTranslateOn = False

translator = Translator()  # Initialize the translator module.
pygame.mixer.init()  # Initialize the mixer module.

# Create a mapping between language names and language codes
language_mapping = {name: code for code, name in LANGUAGES.items()}


def get_language_code(language_name):
    return language_mapping.get(language_name, language_name)


def translator_function(spoken_text, from_language, to_language):
    return translator.translate(spoken_text, src=from_language, dest=to_language)


def text_to_voice(text_data, to_language):
    myobj = gTTS(text=text_data, lang=to_language, slow=False)
    myobj.save("cache_file.mp3")
    audio = pygame.mixer.Sound("cache_file.mp3")  # Load a sound.
    audio.play()
    os.remove("cache_file.mp3")


def main_process(output_placeholder, spoken_text_placeholder, translated_text_placeholder, from_language, to_language):
    global isTranslateOn

    while isTranslateOn:
        rec = sr.Recognizer()
        with sr.Microphone() as source:
            output_placeholder.text("Listening...")
            print("listenning")
            rec.pause_threshold = 1
            audio = rec.listen(source, phrase_time_limit=10)

        try:

            output_placeholder.text("Processing...")
            spoken_text = rec.recognize_google(audio, language=from_language)
            st.session_state.total_text += spoken_text + " "  # Accumulate the spoken text
            # Display spoken text on the screen
            spoken_text_placeholder.text(f"Spoken text: {spoken_text}")

            output_placeholder.text("Translating...")
            translated_text = translator_function(spoken_text, from_language, to_language)

            # Display translated text on the screen
            translated_text_placeholder.text(f"Translated text: {translated_text.text}")

            text_to_voice(translated_text.text, to_language)

        except Exception as e:
            output_placeholder.text(f"Error: {e}")


# UI layout
st.title("Lecture Translation and Summarization")

# Dropdowns for selecting languages
from_language_name = st.selectbox(
    "Select Source Language:", list(LANGUAGES.values()))
to_language_name = st.selectbox(
    "Select Target Language:", list(LANGUAGES.values()))

# Convert language names to language codes
from_language = get_language_code(from_language_name)
to_language = get_language_code(to_language_name)

# Button to trigger translation
start_button = st.button("Start")
stop_button = st.button("Stop")

# Placeholders for dynamic content
output_placeholder = st.empty()  # For status messages (listening, processing, etc.)
spoken_text_placeholder = st.empty()  # For displaying spoken text
translated_text_placeholder = st.empty()  # For displaying translated text
total_text_placeholder = st.empty()  # For displaying total text
summary_placeholder = st.empty()  # For displaying summarized text

# Check if "Start" button is clicked
if start_button:
    if not isTranslateOn:
        isTranslateOn = True
        st.session_state.total_text = ""  # Reset the total text when starting
        main_process(output_placeholder, spoken_text_placeholder, translated_text_placeholder, from_language,
                     to_language)

# Check if "Stop" button is clicked
if stop_button:
    isTranslateOn = False

    new_text='''Since deep learning and machine learning tend to be used interchangeably, it’s worth noting the nuances between the two. Machine learning, deep learning, and neural networks are all sub-fields of artificial intelligence. However, neural networks is actually a sub-field of machine learning, and deep learning is a sub-field of neural networks.

The way in which deep learning and machine learning differ is in how each algorithm learns. "Deep" machine learning can use labeled datasets, also known as supervised learning, to inform its algorithm, but it doesn’t necessarily require a labeled dataset. The deep learning process can ingest unstructured data in its raw form (e.g., text or images), and it can automatically determine the set of features which distinguish different categories of data from one another. This eliminates some of the human intervention required and enables the use of large amounts of data. You can think of deep learning as "scalable machine learning" as Lex Fridman notes in this MIT lecture (link resides outside ibm.com)1.

Classical, or "non-deep," machine learning is more dependent on human intervention to learn. Human experts determine the set of features to understand the differences between data inputs, usually requiring more structured data to learn.

Neural networks, or artificial neural networks (ANNs), are comprised of node layers, containing an input layer, one or more hidden layers, and an output layer. Each node, or artificial neuron, connects to another and has an associated weight and threshold. If the output of any individual node is above the specified threshold value, that node is activated, sending data to the next layer of the network. Otherwise, no data is passed along to the next layer of the network by that node. The “deep” in deep learning is just referring to the number of layers in a neural network. A neural network that consists of more than three layers—which would be inclusive of the input and the output—can be considered a deep learning algorithm or a deep neural network. A neural network that only has three layers is just a basic neural network.

Deep learning and neural networks are credited with accelerating progress in areas such as computer vision, natural language processing, and speech recognition.

See the blog post “AI vs. Machine Learning vs. Deep Learning vs. Neural Networks: What’s the Difference?” for a closer look at how the different concepts relate.

Related content
Explore the watsonx.ai interactive demo

Download "Machine learning for Dummies"

- This link downloads a pdf
Explore Gen AI for developers

'''
    # total_text_placeholder.text_area(new_text)

    # summary_placeholder.text_area(summarize_text(new_text))


    # Summarize the total spoken text
    if st.session_state.total_text.strip():
        summarized_text = summarize_text(st.session_state.total_text)
        summary_placeholder.text_area("Summarized text:", summarized_text, height=100)

