import streamlit as st

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

def main():
    st.title("Mood Detection from Biometrics")
    st.write("Enter your biometric information to determine your mood.")

    # Input: Heart rate in bpm
    heart_rate = st.number_input("Enter your heart rate (bpm):", min_value=30, max_value=200, value=70, step=1)

    if st.button("Determine Mood"):
        # Create a dictionary of biometric metrics.
        bio_metrics = {"heart_rate": heart_rate}
        # Determine mood using the heuristic.
        mood = determine_mood(bio_metrics)
        st.success("Detected Mood: " + mood)

if __name__ == "__main__":
    main()
