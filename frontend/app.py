import streamlit as st
from PIL import Image
import requests
from dotenv import load_dotenv
import os

# Set page tab display
st.set_page_config(
   page_title="Analisando e recriando foto de candidatos eleitos no Brasil",
   page_icon= '🖼️',
   layout="wide",
   initial_sidebar_state="expanded",
)

load_dotenv()

url = os.getenv('LOCAL_API_URL') if os.getenv('DEPLOY') == 'LOCAL' else os.getenv('DOCKER_API_URL')

# App title and description
st.header('Analisando e recriando foto de candidatos eleitos no Brasil 📸')
st.markdown('''
            > Disclaimer: this is a data science project designed for academic purposes. Do not take it seriously.
            ''')

st.markdown("---")

### Create a native Streamlit file upload input
st.markdown("### Let's do a simple face recognition 👇")
img_camera_buffer = st.camera_input("Take a picture")

if img_camera_buffer is not None:

  col1, col2 = st.columns(2)

  with col1:
    ### Display the image user uploaded
    st.image(Image.open(img_camera_buffer), caption="Here's the image you uploaded ☝️")

  with col2:
    with st.spinner("Wait for it..."):
      ### Get bytes from the file buffer
      img_bytes = img_camera_buffer.getvalue()

      ### Make request to  API (stream=True to stream response as bytes)
      res = requests.post(url + "/upload_image", files={'img': img_bytes})

      if res.status_code == 200:
        ### Display the image returned by the API
        st.image(res.content, caption="Image returned from API ☝️")
      else:
        st.markdown("**Oops**, something went wrong 😓 Please try again.")
        print(res.status_code, res.content)

      st.write("""<style>[data-testid="stHorizontalBlock"]{text-align:center;align-items:center;}</style>""",unsafe_allow_html=True)
