import streamlit as st
from transformers import pipeline
import numpy as np
import io
import soundfile as sf

def determine_mood(bio_metrics):
    """
    Determine the mood of a person based on their biometrics.
    
    Parameters:
        bio_metrics (dict): Dictionary containing biometric values.
            Expected key: 'heart_rate' (in bpm).
    
    Returns:
        str: A string indicating the detected mood.
    
    Heuristic:
        - Heart rate > 100 bpm: "Stressed/Anxious"
        - Heart rate < 60 bpm: "Calm/Relaxed"
        - Otherwise: "Neutral/Alert"
    """
    heart_rate = bio_metrics.get("heart_rate")
    if heart_rate is None:
        return "Insufficient data to determine mood."
    
    if heart_rate > 100:
        return "Stressed/Anxious"
    elif heart_rate < 60:
        return "Calm/Relaxed"
    else:
        return "Neutral/Alert"

def mood_to_prompt(mood):
    """
    Map a detected mood to a descriptive prompt for music generation.
    """
    if mood == "Stressed/Anxious":
        return "calming ambient music for relaxation and stress relief"
    elif mood == "Calm/Relaxed":
        return "uplifting and energetic music celebrating calmness"
    elif mood == "Neutral/Alert":
        return "balanced and rhythmic instrumental music"
    else:
        return "soft instrumental music"

def main():
    st.title("Mood-Based Music Generator")
    st.write("Enter your heart rate to determine your mood and generate personalized music.")

    # Collect biometric input
    heart_rate = st.number_input("Enter your heart rate (bpm):", min_value=30, max_value=200, value=70, step=1)

    if st.button("Generate Music"):
        # Determine mood
        bio_metrics = {"heart_rate": heart_rate}
        mood = determine_mood(bio_metrics)
        st.write("Detected Mood:", mood)

        # Map mood to a text prompt
        prompt = mood_to_prompt(mood)
        st.write("Generating music with prompt:", prompt)

        # Load the MusicGen pipeline (this downloads the model on first run)
        pipe = pipeline("text-to-audio", model="facebook/musicgen-small")

        # Generate music from the text prompt
        output = pipe(prompt)
        
        # MusicGen returns a list of dictionaries; take the first result.
        result = output[0]
        audio_array = result["audio"]      # numpy array of audio samples
        sample_rate = result["sample_rate"]  # sample rate (e.g., 44100)

        # Convert the numpy array to WAV bytes using soundfile
        buffer = io.BytesIO()
        sf.write(buffer, audio_array, sample_rate, format="WAV")
        buffer.seek(0)

        # Play the generated audio in the Streamlit app
        st.audio(buffer, format="audio/wav")
        st.success("Music generated successfully!")

if __name__ == "__main__":
    main()
