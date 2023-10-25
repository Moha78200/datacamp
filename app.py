import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import io

# Function to load the pre-trained model
def load_model():
    return tf.keras.models.load_model('model1.hdf5')

# Function to make predictions
def make_predictions(model, uploaded_file):
    try:
        # Read the uploaded file as bytes
        image_bytes = io.BytesIO(uploaded_file.read())
        image = cv2.imdecode(np.frombuffer(image_bytes.read(), np.uint8), -1)

        if image is not None:
            # Resize the image to match the model's input shape
            image = cv2.resize(image, (356, 536))
            image = np.expand_dims(image, axis=0)
            predictions = model.predict(image)
            st.write(predictions)
            
            # Define the threshold for binary classification
            threshold = 0.235
            binary_predictions = [1 if score >= threshold else 0 for score in predictions[0]]

            return binary_predictions
        else:
            st.write("Error: Unable to read the uploaded image.")
            return None
    except Exception as e:
        st.write(f"Error: {e}")
        return None

# Streamlit UI elements and logic
def main():
    st.title("Retinal Disease Classification")

    # Load the model
    model = load_model()

    # File upload
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Make predictions
        binary_predictions = make_predictions(model, uploaded_file)
        if binary_predictions is not None:
            # Display the image
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

            # Display predictions
            st.write("Predictions:")
            
            # List of class labels
            class_labels = [
                'Disease_Risk', 'DR', 'ARMD', 'MH', 'DN', 'MYA', 'BRVO', 'TSLN', 'ERM', 'LS',
                'MS', 'CSR', 'ODC', 'CRVO', 'TV', 'AH', 'ODP', 'ODE', 'ST', 'AION', 'PT',
                'RT', 'RS', 'CRS', 'EDN', 'RPEC', 'MHL', 'RP', 'CWS', 'CB', 'ODPM', 'PRH',
                'MNF', 'HR', 'CRAO', 'TD', 'CME', 'PTCR', 'CF', 'VH', 'MCA', 'VS', 'BRAO',
                'PLQ', 'HPED', 'CL'
            ]

            # Display predictions as 0 or 1 for each disease
            for label, prediction in zip(class_labels, binary_predictions):
                st.write(f"{label}: {prediction}")
            

if __name__ == '__main__':
    st.set_option('deprecation.showfileUploaderEncoding', False)


    main()
