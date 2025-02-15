import streamlit as st
import io
import soundfile as sf
from transformers import AutoProcessor, MusicgenForConditionalGeneration

def determine_mood(bio_metrics):
    """
    Determine the mood based on heart rate.
    Heuristic:
      - > 100 bpm: "Stressed/Anxious"
      - < 60 bpm: "Calm/Relaxed"
      - Otherwise: "Neutral/Alert"
    """
    heart_rate = bio_metrics.get("heart_rate")
    if heart_rate is None:
        return "Insufficient data"
    if heart_rate > 100:
        return "Stressed/Anxious"
    elif heart_rate < 60:
        return "Calm/Relaxed"
    else:
        return "Neutral/Alert"

def mood_to_prompt(mood):
    """
    Map the detected mood to a descriptive text prompt for music generation.
    """
    if mood == "Stressed/Anxious":
        return "calming ambient music for relaxation and stress relief"
    elif mood == "Calm/Relaxed":
        return "uplifting and soothing lo-fi music with a gentle beat"
    elif mood == "Neutral/Alert":
        return "balanced instrumental music with a steady rhythm"
    else:
        return "soft instrumental background music"

def main():
    st.title("Mood-Based Music Generator with MusicGen")
    st.write("Enter your heart rate to determine your mood and generate personalized music.")

    # Input: Heart rate in bpm
    heart_rate = st.number_input("Enter your heart rate (bpm):", min_value=30, max_value=200, value=70, step=1)
    
    if st.button("Generate Music"):
        # Determine the mood
        bio_metrics = {"heart_rate": heart_rate}
        mood = determine_mood(bio_metrics)
        st.write("Detected Mood:", mood)
        
        # Map mood to a text prompt for MusicGen
        prompt = mood_to_prompt(mood)
        st.write("Generating music for prompt:", prompt)
        
        with st.spinner("Loading MusicGen model and generating music..."):
            # Load processor and model (this may take a moment)
            processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
            model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
            
            # Process the text prompt; wrap it in a list for batch processing.
            inputs = processor(
                text=[prompt],
                padding=True,
                return_tensors="pt",
            )
            
            # Generate audio; adjust max_new_tokens if needed.
            audio_values = model.generate(**inputs, max_new_tokens=256)
            sampling_rate = model.config.audio_encoder.sampling_rate
            
            # The generated tensor is expected to have shape (batch_size, channels, samples).
            # We'll use the first channel of the first batch element.
            audio_data = audio_values[0, 0].cpu().numpy()
            
            # Write the audio data to an in-memory WAV file.
            buffer = io.BytesIO()
            sf.write(buffer, audio_data, sampling_rate, format="WAV")
            buffer.seek(0)
        
        # Play the generated audio.
        st.audio(buffer, format="audio/wav")
        st.success("Music generated successfully!")

if __name__ == "__main__":
    main()
