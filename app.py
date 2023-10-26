import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import io
import pandas as pd
from PIL import Image
# Function to load the trained model
@st.cache_data(show_spinner = "Loading model...")
def load_model():
    return tf.keras.models.load_model('model1.hdf5')

# Function to make predictions
@st.cache_data(show_spinner = "Making predictions...")
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
            #st.write(predictions)
            
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
@st.cache_data(experimental_allow_widgets=True)
def main():
    with st.container():
        st.title("Retinal Disease Classification")
        st.write("\nOur website allows you to analyse your medical X-ray of the eye and determine whether it has cancer or not. If so, we'll tell you which one(s).")
        st.subheader("Analysis of your X-Ray")
        st.write("\nAll you need to do is upload your file below in .jpg, .png or .jpeg format and we will take care of the rest.")
        # Load the model
        model = load_model()

        # File upload
        uploaded_file = st.file_uploader("Upload your X-Ray...", type=["jpg", "png", "jpeg"])
        count=0
        if uploaded_file is not None:
            # Make predictions
            binary_predictions = make_predictions(model, uploaded_file)
            if binary_predictions is not None:
                # Display the image
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

                # Display predictions
                st.write("Result of our analysis:")

                # List of class labels
                class_labels = [
                    'Disease_Risk', 'DR', 'ARMD', 'MH', 'DN', 'MYA', 'BRVO', 'TSLN', 'ERM', 'LS',
                    'MS', 'CSR', 'ODC', 'CRVO', 'TV', 'AH', 'ODP', 'ODE', 'ST', 'AION', 'PT',
                    'RT', 'RS', 'CRS', 'EDN', 'RPEC', 'MHL', 'RP', 'CWS', 'CB', 'ODPM', 'PRH',
                    'MNF', 'HR', 'CRAO', 'TD', 'CME', 'PTCR', 'CF', 'VH', 'MCA', 'VS', 'BRAO',
                    'PLQ', 'HPED', 'CL'
                ]

                # Display predictions as 0 or 1 for each disease
                # Dictionnaire avec texte personnalisé par maladie
                texts = {
                    'Disease_Risk': "⚠ Unfortunately, we detect the presence of cancer.",
                    'DR': "➪ You are at risk of diabetic retinopathy.",
                    'ARMD': "➪ Based on the results, you show signs of age-related macular degeneration.",
                    'MH': "➪ Vitreous opacities have been detected.",
                    'DN': "➪ Drusen have been identified.",
                    'MYA': "➪ You show signs of myopia.",
                    'BRVO': "➪ It seems you have a branch retinal vein occlusion.",
                    'TSLN': "➪ Tessellated lines have been observed.",
                    'ERM': "➪ An epiretinal membrane has been detected.",
                    'LS': "➪ Laser scars are visible.",
                    'MS': "➪ Macular scars are present.",
                    'CSR': "➪ The results suggest central serous retinopathy.",
                    'ODC': "➪ The cupping of the optic disc seems important.",
                    'CRVO': "➪ This is probably a central retinal vein occlusion.",
                    'TV': "➪ Tortuous retinal vessels have been noted.",
                    'AH': "➪ Asteroid hyalosis has been detected.",
                    'ODP': "➪ The optic disc appears pale.",
                    'ODE': "➪ Optic disc edema is visible.",
                    'ST': "➪ An optociliary shunt has been identified.",
                    'AION': "➪ The results suggest anterior ischemic optic neuropathy.",
                    'PT': "➪ Parafoveal telangiectasias are present.",
                    'RT': "➪ Retinal traction appears to be present.",
                    'RS': "➪ Signs of retinitis have been detected.",
                    'CRS': "➪ This could be choroidoretinitis.",
                    'EDN': "➪ Retinal exudation is visible.",
                    'RPEC': "➪ Changes in the retinal pigment epithelium have been noted.",
                    'MHL': "➪ A macular hole is suspected.",
                    'RP': "➪ The symptoms correspond to retinitis pigmentosa.",
                    'CWS': "➪ Cotton wool spots are visible.",
                    'CB': "➪ A coloboma is present.",
                    'ODPM': "➪ Optic disc pit maculopathy has been diagnosed.",
                    'PRH': "➪ A preretinal hemorrhage has been detected.",
                    'MNF': "➪ Myelinated nerve fibers have been observed.",
                    'HR': "➪ A hemorrhagic retinopathy is suspected.",
                    'CRAO': "➪ This is probably a central retinal artery occlusion.",
                    'TD': "➪ A tilted optic disc has been noted.",
                    'CME': "➪ Cystoid macular edema has been detected.",
                    'PTCR': "➪ A post-traumatic choroidal rupture has been diagnosed.",
                    'CF': "➪ Choroidal folds are visible.",
                    'VH': "➪ A vitreous hemorrhage is present.",
                    'MCA': "➪ A macroaneurysm has been detected.",
                    'VS': "➪ Signs of vasculitis are observed.",
                    'BRAO': "➪ This could be a branch retinal artery occlusion.",
                    'PLQ': "➪ A plaque is visible.",
                    'HPED': "➪ A hemorrhagic pigment epithelial detachment has been diagnosed.",
                    'CL': "➪ Vascular collaterals are present."
                }

                df = pd.DataFrame(columns=['Disease', 'Result'])

                for label, prediction in zip(class_labels, binary_predictions):
                    df.loc[len(df)] = [label, prediction]

                st.table(df)
                # Boucle sur les lignes pour personnaliser
                for index, row in df.iterrows():
                    if row['Result'] == 1:
                        st.write(texts[row['Disease']])
                        count+=1
                    if row['Result']==0 and row['Disease']=='Disease_Risk':
                        st.write("Congratulations, there's absolutely nothing wrong with you, your eye is in good health 👍👍 !!!")

                if count>0:
                    st.write("⚠ Consult a doctor as soon as possible.")
                    link = f"<a href='https://www.fo-rothschild.fr/patient/loffre-de-soins/retine-medicale' target='_blank'>web site</a>"

                    st.markdown(f"If you're not sure what to do, don't hesitate to go to this {link} and follow the recommended protocol.",
                                unsafe_allow_html=True)

    st.markdown("""
    <style>
    div.block-container{
       max-width: 60% !important;
    }
    </style>  
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    st.set_option('deprecation.showfileUploaderEncoding', False)
    main()
