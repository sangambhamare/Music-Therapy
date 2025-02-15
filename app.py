import streamlit as st
import os
import tempfile
import urllib.request
import note_seq
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from magenta.models.melody_rnn import melody_rnn_sequence_generator
from magenta.models.shared import sequence_generator_bundle
from magenta.protobuf import generator_pb2

# Cache the model loading to speed up subsequent runs.
@st.cache(allow_output_mutation=True)
def load_model():
    bundle_path = 'basic_rnn.mag'
    if not os.path.exists(bundle_path):
        st.info("Downloading pre-trained model bundle...")
        url = 'https://storage.googleapis.com/magentadata/models/melody_rnn/basic_rnn.mag'
        urllib.request.urlretrieve(url, bundle_path)
    bundle = sequence_generator_bundle.read_bundle_file(bundle_path)
    generator_map = melody_rnn_sequence_generator.get_generator_map()
    model = generator_map['basic_rnn'](checkpoint=None, bundle=bundle)
    model.initialize()
    return model

def get_music_params_from_heart_rate(heart_rate):
    """
    Map heart rate to music generation parameters.
    """
    if heart_rate > 100:
        temperature = 0.5      # Calmer music
        primer_pitch = 60      # Middle C
        gen_end_time = 30      # Longer piece for sustained therapy
    elif heart_rate < 60:
        temperature = 1.2      # More variation
        primer_pitch = 72      # Brighter note (C5)
        gen_end_time = 20      # Shorter piece
    else:
        temperature = 0.8      # Balanced parameters
        primer_pitch = 64      # E note
        gen_end_time = 25
    return temperature, primer_pitch, gen_end_time

def generate_music(model, heart_rate):
    # Map the heart rate to generation parameters.
    temperature, primer_pitch, gen_end_time = get_music_params_from_heart_rate(heart_rate)
    st.write(f"Using parameters: Temperature = {temperature}, Primer Pitch = {primer_pitch}, Generation End Time = {gen_end_time}s")
    
    # Create a primer NoteSequence (seed note).
    primer_sequence = note_seq.NoteSequence()
    primer_sequence.notes.add(pitch=primer_pitch, start_time=0.0, end_time=0.5, velocity=80)
    primer_sequence.total_time = 1.0

    # Configure generation options.
    generator_options = generator_pb2.GeneratorOptions()
    generator_options.generate_sections.add(
        start_time=primer_sequence.total_time,
        end_time=gen_end_time
    )
    generator_options.args['temperature'].float_value = temperature

    # Generate the music.
    generated_sequence = model.generate(primer_sequence, generator_options)
    return generated_sequence

def main():
    st.title("Music Therapy Generator")
    st.write("Enter your biometric data to generate personalized music.")

    # Input field for the user's heart rate.
    heart_rate = st.number_input("Enter your Heart Rate (bpm):", min_value=30, max_value=200, value=70, step=1)

    if st.button("Generate Music"):
        model = load_model()
        st.write("Generating your personalized music...")
        generated_sequence = generate_music(model, heart_rate)

        # Save the generated NoteSequence as a MIDI file in a temporary file.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as tmp_file:
            midi_path = tmp_file.name
        note_seq.sequence_proto_to_midi_file(generated_sequence, midi_path)

        st.success("Music generation complete!")
        # Provide a download button for the MIDI file.
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
