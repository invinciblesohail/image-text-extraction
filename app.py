from google import genai

from google.genai import types
from google.api_core import retry
import cv2
import matplotlib.pyplot as plt
import io
import os
import base64
import tempfile
from PIL import Image
import streamlit as st
import numpy as np
from dotenv import load_dotenv

@st.cache_resource
def load_model():
    is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})

    genai.models.Models.generaete_content = retry.Retry(predicate=is_retriable)(genai.models.Models.generate_content)

    
    load_dotenv()
    #api_key  = os.environ.get("GOOGLE_API_KEY")
    api_key = st.secrets["GOOGLE_API_KEY"]
    
    client = genai.Client(api_key=api_key)
    
    return client

client = load_model()

def enlarge_image(original, scale_factor=2.0, interpolation_method=cv2.INTER_CUBIC):
    """
    Enlarge an image and display comparison 
    :param image_path: Path to input image
    :param scale_factor: Multiplier for enlargement (e.g., 2.0 = 200%)
    :param interpolation_method: cv2.INTER_LINEAR, INTER_CUBIC, INTER_LANCZOS4

    """
    #original = cv2.imread(image_path)
    if original is None:
        print('Error: Could not read image')
        return 
    new_width = int(original.shape[1] * scale_factor)
    new_height = int(original.shape[0] * scale_factor)

    enlarged = cv2.resize(original, (new_width, new_height))

    return enlarged

def main():
    try:
        st.title("Text Extractor with Gemini AI")
        st.markdown("Upload an image containing mathematical equations and get them extracted by AI")

        # File uploader
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Process the image
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
            

            image = cv2.imread(tmp_file_path)
            if image is not None:
                # Display original image
                st.subheader("Original Image")
                st.image(image, channels="BGR", use_container_width=True)
                
                
                enlarged_img = enlarge_image(image, scale_factor=3.0)
                enlarged_img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                #st.subheader("Enlarged Image")
                #st.image(enlarged_img, channels="BGR", use_container_width=True)
                # Step 2: Convert NumPy array to PIL Image
                #_, file_extension = os.path.splitext(image_path)
                #image_format = os.path.splitext(uploaded_file.name)[1][1:].lower()
                # = file_extension[1:].lower()
                pil_image = Image.fromarray(enlarged_img_rgb)


                buffer = io.BytesIO()
                pil_image.save(buffer, format='png')
                image_bytes = buffer.getvalue()
                image_b64 = base64.b64encode(image_bytes).decode('utf-8')
                model_name = "gemini-2.0-flash"
                model_config = types.GenerateContentConfig(temperature=0.1, top_k=1, 
                                            max_output_tokens=500)

                if st.button("Extract Text"):
                    with st.spinner("Analyzing with Gemini AI..."):
                        contents = [
                            {
                                "role": "user",
                                "parts": [
                                    {"text": "Extract all the text from this image:"},
                                    {
                                        "inline_data": {
                                            "mime_type": f"image/png",
                                            "data": image_b64
                                        }
                                    }
                                ]
                            }
                        ]
                        response = client.models.generate_content(
                                model = model_name, config= model_config, contents=contents
                            )
                        buf = io.StringIO()
                        buf.write(response.text)
                        
                        st.subheader("Extracted Text")
                        
                        st.markdown(buf.getvalue())
                        st.success("Text Extraction completed Successfully!")
                        
                else:
                    st.error("Failed to process image for Gemini AI")

        else:
            st.error("Failed to read the uploaded image")
    except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    
if __name__ == "__main__":
    main()
