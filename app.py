import streamlit as st
import numpy as np
import os
import tempfile

# Using TensorFlow's Keras API to load the pretrained DeepJazz model.
from tensorflow.keras.models import load_model
from midiutil import MIDIFile

# ---------------------------------------------------
# Helper: Load the pretrained model (DeepJazz)
# ---------------------------------------------------
@st.cache(allow_output_mutation=True)
def load_deepjazz_model():
    model_path = 'deepjazz.h5'
    if not os.path.exists(model_path):
        st.error("Pretrained model file 'deepjazz.h5' not found. Please add it to your repository.")
        return None
    model = load_model(model_path)
    return model

# ---------------------------------------------------
# Helper: Map heart rate to generation parameters.
# ---------------------------------------------------
def get_music_params_from_heart_rate(heart_rate):
    """
    For demonstration purposes, a higher heart rate returns a lower temperature (calmer output)
    and a longer generated sequence, while a lower heart rate returns a higher temperature.
    """
    if heart_rate > 100:
        temperature = 0.5  # Less randomness = calmer music
        sequence_length = 100  # More notes (longer piece)
    elif heart_rate < 60:
        temperature = 1.2  # More randomness
        sequence_length = 50   # Fewer notes
    else:
        temperature = 0.8
        sequence_length = 75
    return temperature, sequence_length

# ---------------------------------------------------
# Helper: Sample a note index using temperature.
# ---------------------------------------------------
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-7) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# ---------------------------------------------------
# Helper: Generate music using the pretrained model.
# ---------------------------------------------------
def generate_music(model, temperature, sequence_length, seed=None):
    """
    This function assumes:
      - The model was trained on sequences of length 'maxlen'
      - The output of the model is a probability distribution over a vocabulary (here assumed to be 128 MIDI notes)
    """
    maxlen = 50   # Input sequence length expected by the model
    vocab_size = 128  # MIDI note range 0-127

    # If no seed is provided, generate a random seed sequence.
    if seed is None:
        seed = np.random.randint(0, vocab_size, size=(maxlen,))
    
    generated = list(seed)
    
    # Generate notes one at a time.
    for i in range(sequence_length):
        # Prepare one-hot encoded input of shape (1, maxlen, vocab_size)
        x_pred = np.zeros((1, maxlen, vocab_size))
        for t, note in enumerate(seed):
            x_pred[0, t, note] = 1.0
        
        # Predict next note distribution.
        preds = model.predict(x_pred, verbose=0)[0]  # Expected shape: (vocab_size,)
        next_index = sample(preds, temperature)
        
        generated.append(next_index)
        # Slide the seed window forward.
        seed = np.append(seed[1:], next_index)
    
    return generated

# ---------------------------------------------------
# Helper: Convert a note sequence to a MIDI file.
# ---------------------------------------------------
def sequence_to_midi(note_sequence, tempo=120):
    midi = MIDIFile(1)  # One track MIDI file
    track = 0
    time = 0  # Start time in beats
    midi.addTempo(track, time, tempo)
    
    channel = 0
    duration = 1  # Duration (in beats) for each note
    volume = 100  # Volume for each note
    for note in note_sequence:
        # Only add valid MIDI notes.
        if 0 <= note < 128:
            midi.addNote(track, channel, note, time, duration, volume)
            time += duration
    return midi

def save_midi(midi, file_path):
    with open(file_path, "wb") as output_file:
        midi.writeFile(output_file)

# ---------------------------------------------------
# Main Streamlit App
# ---------------------------------------------------
def main():
    st.title("DeepJazz Music Generator")
    st.write("Generate personalized jazz music using a pretrained DeepJazz model. "
             "Your heart rate input will adjust the generation parameters.")
    
    # Input: Manual biometric (heart rate) entry.
    heart_rate = st.number_input("Enter your Heart Rate (bpm):", min_value=30, max_value=200, value=70, step=1)
    
    if st.button("Generate Music"):
        model = load_deepjazz_model()
        if model is None:
            return
        
        temperature, sequence_length = get_music_params_from_heart_rate(heart_rate)
        st.write(f"Using parameters: Temperature = {temperature}, Sequence Length = {sequence_length} notes")
        
        with st.spinner("Generating music..."):
            generated_sequence = generate_music(model, temperature, sequence_length)
        
        # Convert generated note sequence to MIDI.
        midi = sequence_to_midi(generated_sequence)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as tmp_file:
            midi_path = tmp_file.name
        save_midi(midi, midi_path)
        
        st.success("Music generated successfully!")
        with open(midi_path, "rb") as f:
            midi_bytes = f.read()
        st.download_button(
            label="Download MIDI File",
            data=midi_bytes,
            file_name="generated_music.mid",
            mime="audio/midi"
        )

if __name__ == "__main__":
    main()
