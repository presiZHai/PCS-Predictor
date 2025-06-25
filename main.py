import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Load Pre-trained Model and Feature Names ---
# This function will be cached by Streamlit, so the model is only loaded once.
@st.cache_resource
def load_model():
    """
    Loads the pre-trained RandomForestClassifier model and the list of feature names.
    Assumes 'final_model.joblib' and 'feature_names.joblib' are in the same directory.
    """
    try:
        model = joblib.load('final_model.joblib')
        feature_names = joblib.load('feature_names.joblib')
        return model, feature_names
    except FileNotFoundError:
        return None, None

def main():
    """
    Main function to run the Streamlit prediction application.
    """
    st.set_page_config(page_title="HIV Client Satisfaction Predictor", layout="wide")
    st.title("Client Satisfaction Prediction for HIV Care")

    # Load the model and feature names
    model, feature_names = load_model()

    if model is None or feature_names is None:
        st.error(
            "Model files not found. Please ensure 'final_model.joblib' and "
            "'feature_names.joblib' are present in the application directory. "
            "You need to run the training script first to generate these files."
        )
        st.stop()

    st.info("Please rate the following aspects of the service on a scale from 'Strongly Disagree' to 'Strongly Agree'.")

    # --- User Input Interface ---
    # Define the Likert scale options and their corresponding numerical values
    likert_options = ['Strongly Disagree', 'Disagree', 'Neither Agree or Disagree', 'Agree', 'Strongly Agree']
    likert_mapping = {option: i + 1 for i, option in enumerate(likert_options)}

    # Create two columns for the input fields
    col1, col2 = st.columns(2)

    user_inputs = {}
    # Distribute the input fields across the two columns
    mid_point = len(feature_names) // 2
    
    with col1:
        for feature in feature_names[:mid_point]:
            selected_option = st.selectbox(label=feature, options=likert_options, index=2) # Default to 'Neither'
            user_inputs[feature] = likert_mapping[selected_option]

    with col2:
        for feature in feature_names[mid_point:]:
            selected_option = st.selectbox(label=feature, options=likert_options, index=2)
            user_inputs[feature] = likert_mapping[selected_option]


    # --- Prediction Logic ---
    if st.button('Predict Satisfaction', type="primary"):
        # Create a DataFrame from the user's input in the correct feature order
        input_df = pd.DataFrame([user_inputs])
        input_df = input_df[feature_names] # Ensure the columns are in the correct order

        # Make a prediction
        prediction_numeric = model.predict(input_df.values)
        prediction_proba = model.predict_proba(input_df.values)


        # Map the numeric prediction back to a meaningful label
        satisfaction_labels = {
            0: 'Very Dissatisfied',
            1: 'Neutral',
            2: 'Satisfied',
            3: 'Very Satisfied'
        }
        prediction_label = satisfaction_labels.get(prediction_numeric[0], "Unknown")

        # Display the result
        st.success(f"**Predicted Client Satisfaction:** {prediction_label}")

        # Display prediction probability
        st.write("Prediction Confidence:")
        proba_df = pd.DataFrame(
            prediction_proba,
            columns=[satisfaction_labels[i] for i in model.classes_],
            index=['Probability']
        )
        st.dataframe(proba_df.style.format("{:.2%}"))


if __name__ == '__main__':
    main()