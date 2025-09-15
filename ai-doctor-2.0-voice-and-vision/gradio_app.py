import os
import gradio as gr
import pyttsx3  # Offline TTS fallback
from dotenv import load_dotenv

# Importing project modules
from brain_of_the_doctor import encode_image, analyze_image_with_query
from voice_of_the_patient import record_audio, transcribe_with_groq
from voice_of_the_doctor import text_to_speech_with_elevenlabs

# Load environment variables if .env is used
load_dotenv()

# Define AI Doctor's response style
system_prompt = """You have to act as a professional doctor, I know you are not but this is for learning purposes.
            What's in this image? Do you find anything wrong with it medically? 
            If you make a differential, suggest some remedies for them. Do not add any numbers or special characters in 
            your response. Your response should be in one long paragraph. Also always answer as if you are answering a real person.
            Do not say 'In the image I see' but say 'With what I see, I think you have ....'
            Do not respond as an AI model in markdown. Your answer should mimic that of an actual doctor, not an AI bot.
            Keep your answer concise (max 2 sentences). No preamble, start your answer right away, please."""

# Safe Text-to-Speech Function with Fallback
def text_to_speech_safe(input_text, output_filepath="final.mp3"):
    try:
        from gtts import gTTS
        tts = gTTS(input_text)
        tts.save(output_filepath)
        return output_filepath
    except Exception as e:
        print(f"gTTS failed: {e}. Falling back to offline TTS.")
        engine = pyttsx3.init()
        engine.save_to_file(input_text, output_filepath)
        engine.runAndWait()
        return output_filepath

# Processing Function
def process_inputs(audio_filepath, image_filepath):
    try:
        # Transcribe audio
        speech_to_text_output = transcribe_with_groq(
            GROQ_API_KEY=os.environ.get("GROQ_API_KEY"),
            audio_filepath=audio_filepath,
            stt_model="whisper-large-v3"
        )
    except Exception as e:
        print(f"Speech-to-text failed: {e}")
        speech_to_text_output = "Error: Could not transcribe audio."

    # Handle image input
    doctor_response = "No image provided for analysis."
    if image_filepath:
        try:
            doctor_response = analyze_image_with_query(
                query=system_prompt + speech_to_text_output,
                encoded_image=encode_image(image_filepath),
                model="llama-3.2-11b-vision-preview"
            )
        except Exception as e:
            print(f"Image analysis failed: {e}")
            doctor_response = "Error: Could not analyze the image."

    # Generate audio response (Use ElevenLabs if available, otherwise fallback)
    try:
        voice_of_doctor = text_to_speech_with_elevenlabs(input_text=doctor_response, output_filepath="final.mp3")
    except Exception:
        voice_of_doctor = text_to_speech_safe(doctor_response)

    return speech_to_text_output, doctor_response, voice_of_doctor

# Create the Gradio Interface
iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath", label="Record Your Symptoms"),
        gr.Image(type="filepath", label="Upload Medical Image (Optional)")
    ],
    outputs=[
        gr.Textbox(label="Speech to Text"),
        gr.Textbox(label="Doctor's Response"),
        gr.Audio(label="Doctor's Voice Response")
    ],
    title="AI Doctor with Vision and Voice",
    description="Speak your symptoms and upload a medical image to get AI-generated medical insights."
)

# Launch Gradio App
iface.launch(debug=True)
