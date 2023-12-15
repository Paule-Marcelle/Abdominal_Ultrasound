def config():
        original_title = '<h1 style="font-family: serif; color:white; font-size: 20px;"></h1>'
        st.markdown(original_title, unsafe_allow_html=True)
        # Set the background image
        background_image = """
        <style>
        [data-testid="stAppViewContainer"] > .main {
            background-image: url("https://img.freepik.com/vecteurs-libre/design-fond-abstrait-technologie-donnees-volumineuses_1017-22911.jpg?w=826&t=st=1702595736~exp=1702596336~hmac=e6ec68887211c57605a3f4314790d4e083211903430fb21566465b3c67bbaba9");
            background-size: 100vw 100vh;  # This sets the size to cover 100% of the viewport width and height
            background-position: center;
            background-repeat: no-repeat;
        }
        </style>
        """
        st.markdown(background_image, unsafe_allow_html=True)
