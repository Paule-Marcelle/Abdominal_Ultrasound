import streamlit as st
def config():
        original_title = '<h1 style="font-family: serif; color:white; font-size: 30px;"></h1>'
        st.markdown(original_title, unsafe_allow_html=True)

        #Définissez les paramètres de style en tant que variables dans la fonction
        # font_size = "30px"
        # font_family = "Times New Roman, serif"
        # text_color = "white"
        # return font_size, font_family, text_color


        # Set the background image
        background_image = """
        <style>
        [data-testid="stAppViewContainer"] > .main {
            background-image: url("https://static9.depositphotos.com/1006214/1185/i/450/depositphotos_11859178-stock-photo-nurse-holding-table-computer.jpg");
            background-size: 100vw 100vh;  # This sets the size to cover 100% of the viewport width and height
            background-position: center;
            background-repeat: no-repeat;
        }
        </style>
        """
        st.markdown(background_image, unsafe_allow_html=True)
