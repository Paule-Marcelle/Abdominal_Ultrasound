import streamlit as st
import tensorflow as tf
from logging import PlaceHolder
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout,BatchNormalization,Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import streamlit  # 👈 Add the caching decorator
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from util import config

# Définir la taille de la police souhaitée
font_size = "30px"

# Définir la police de caractères souhaitée
font_family = "Times New Roman, serif"


# Définir la couleur du texte souhaitée (code hexadécimal)
text_color = "green"

# Custom style settings
button_style = f"font-size: 50px; color: white; background-color: #2e6b8e; border-radius: 8px; padding: 10px 20px;"

result_text_style = f"font-size: 30px; color: white; font-family: 'Times New Roman', serif;"

config()

# Charger le modèle
model = load_model('model_US.h5')

# Fonction de prédiction
def predict(img):
    # Redimensionner et convertir l'image en RGB
    img = img.resize((64, 64)).convert('RGB')
    # Prétraiter l'image
    img = img.resize((64, 64))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normaliser les valeurs des pixels

    # Effectuer la prédiction
    prediction = model.predict(img)
    return prediction

def main():
    # Définir la taille de la police souhaitée
    font_size = "30px"

# Définir la police de caractères souhaitée
    font_family = "Times New Roman, serif"

# Définir la couleur du texte souhaitée (code hexadécimal)
    text_color = "yellow"

    st.title("Application de Prediction de Grossesse grâce aux Echographies Abdominales")
    # Widget pour télécharger une image
    uploaded_file = st.file_uploader("Uploader une image d'échographie abdominale...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Affichez l'image téléchargée
        image = Image.open(uploaded_file)
        st.image(image, caption="Image téléchargée", use_column_width=True)

        # Bouton de prédiction
        if st.button("PREDICTION"):
            # Effectuez la prédiction en utilisant votre fonction predict
            prediction_result = predict(image)
            predicted_labels = np.where(prediction_result > 0.5, 1, 0)

            # Interpret the prediction
            probability = predicted_labels[0][0]
            if probability >= 0.5 :
                    
                detection = "PREGNANCY"
            else:
                detection = "NO PREGNANCY"



    # Utiliser les paramètres de style dans la chaîne HTML
            st.markdown(
                f'<div style="font-size: {font_size}; color: {text_color}; font-family: {font_family};">'
                '</div>',
                unsafe_allow_html=True
            )
       
            st.write(
                    f'<div style="font-size: {font_size}; color: {text_color}; font-family: {font_family};">'
                    f"Diagnostic: {detection} (Probability: {probability:.3f})"
                    '</div>',
                    unsafe_allow_html=True
                )   

if __name__ == "__main__":
    main()

