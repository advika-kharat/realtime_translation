import os
import pygame
from gtts import gTTS
import streamlit as st
import speech_recognition as sr
from googletrans import LANGUAGES, Translator
from transformers import pipeline  # For summarization
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas  # For PDF creation
from io import BytesIO  # For in-memory PDF
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet  # For better text formatting

isTranslateOn = False

translator = Translator()  # Initialize the translator module.
pygame.mixer.init()  # Initialize the mixer module.
summarizer = pipeline("summarization")  # Initialize the summarization model

# Create a mapping between language names and language codes
language_mapping = {name: code for code, name in LANGUAGES.items()}


def get_language_code(language_name):
    return language_mapping.get(language_name, language_name)


def translator_function(spoken_text, from_language, to_language):
    return translator.translate(spoken_text, src=from_language, dest=to_language)


def text_to_voice(text_data, to_language):
    try:
        myobj = gTTS(text=text_data, lang=to_language, slow=False)
        myobj.save("cache_file.mp3")
        audio = pygame.mixer.Sound("cache_file.mp3")  # Load a sound.
        audio.play()
        os.remove("cache_file.mp3")
    except Exception as e:
        print(f"Error in text_to_voice: {str(e)}")


def generate_summary(text_data):
    # Use the summarizer pipeline to generate a summary of the translated text
    summarized = summarizer(text_data, max_length=50, min_length=25, do_sample=False)
    return summarized[0]['summary_text']


def create_pdf(summary_text):
    # Create a PDF from the summary
    pdf_buffer = BytesIO()

    # Create a PDF using SimpleDocTemplate and Paragraph for proper text formatting and wrapping
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)

    # Use a stylesheet for styling the text
    styles = getSampleStyleSheet()
    style_normal = styles['Normal']

    # Paragraph with summary text, ensuring wrapping
    summary_paragraph = Paragraph(summary_text, style_normal)

    # Build the PDF with the paragraph
    doc.build([summary_paragraph])

    # Move buffer position to beginning to send file to user
    pdf_buffer.seek(0)
    return pdf_buffer


def main_process(output_placeholder, spoken_text_placeholder, translated_text_placeholder, summary_placeholder,
                 from_language, to_language):
    global isTranslateOn

    while isTranslateOn:

        rec = sr.Recognizer()
        with sr.Microphone() as source:
            output_placeholder.text("Listening...")
            rec.pause_threshold = 1
            try:
                audio = rec.listen(source, phrase_time_limit=10)
            except Exception as e:
                output_placeholder.text(f"Error while listening: {str(e)}")
                continue

        try:
            output_placeholder.text("Processing...")
            spoken_text = rec.recognize_google(audio, language=from_language)

            # Display spoken text on the screen
            spoken_text_placeholder.text(f"Spoken text: {spoken_text}")

            output_placeholder.text("Translating...")
            translated_text = translator_function(spoken_text, from_language, to_language)

            # Display translated text on the screen
            translated_text_placeholder.text(f"Translated text: {translated_text.text}")

            # Always translate to English for summarization
            translated_to_english = translator_function(translated_text.text, to_language, 'en')
            summary = generate_summary(translated_to_english.text)

            # Display the summary
            summary_placeholder.text(f"Summary (in English): {summary}")

            # Create PDF of the summary
            pdf_file = create_pdf(summary)

            # Offer the PDF file for download
            st.download_button(
                label="Download Summary as PDF",
                data=pdf_file,
                file_name="summary.pdf",
                mime="application/pdf"
            )

            # Play the translated text in the target language
            text_to_voice(translated_text.text, to_language)

        except sr.UnknownValueError:
            output_placeholder.text("Could not understand the audio, please try again.")
        except sr.RequestError as e:
            output_placeholder.text(f"Could not request results from Google Speech Recognition service; {e}")
        except Exception as e:
            output_placeholder.text(f"Error: {str(e)}")


# UI layout
st.title("Language Translator with Summary and PDF")

# Dropdowns for selecting languages
from_language_name = st.selectbox("Select Source Language:", list(LANGUAGES.values()))
to_language_name = st.selectbox("Select Target Language:", list(LANGUAGES.values()))

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
summary_placeholder = st.empty()  # For displaying summary

# Check if "Start" button is clicked
if start_button:
    if not isTranslateOn:
        isTranslateOn = True
        main_process(output_placeholder, spoken_text_placeholder, translated_text_placeholder, summary_placeholder,
                     from_language, to_language)

# Check if "Stop" button is clicked
if stop_button:
    isTranslateOn = False
