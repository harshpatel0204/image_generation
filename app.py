import os
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

# Load environment variables from .env file
load_dotenv()

# Set the Google API key
API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize session state for image
if "generated_image" not in st.session_state:
    st.session_state.generated_image = None

# Streamlit UI
st.title("IMAGE GENERATION APP")
uploaded_file = st.file_uploader("Upload an input image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", width=700)

text_input = st.text_input("Enter your prompt")
generate = st.button("Generate Image")

# Generate image and store in session state
if uploaded_file and text_input and generate:
    with st.spinner("Generating image..."):
        image = Image.open(uploaded_file)

        client = genai.Client(api_key=API_KEY)

        response = client.models.generate_content(
            model="gemini-2.0-flash-preview-image-generation",
            contents=[text_input, image],
            config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
        )

        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                image_bytes = BytesIO(part.inline_data.data)
                generated_image = Image.open(image_bytes)
                st.session_state.generated_image = generated_image

# Display image from session state (after generation or rerun)
if st.session_state.generated_image is not None:
    st.image(
        st.session_state.generated_image,
        caption="Generated Image",
        use_container_width=True,
    )

    # Prepare download
    download_bytes = BytesIO()
    st.session_state.generated_image.save(download_bytes, format="PNG")
    download_bytes.seek(0)

    st.download_button(
        label="Download Image",
        data=download_bytes,
        file_name="generated_image.png",
        mime="image/png",
    )
