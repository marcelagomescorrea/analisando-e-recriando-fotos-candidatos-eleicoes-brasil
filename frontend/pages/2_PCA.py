import streamlit as st
from PIL import Image
import requests
from dotenv import load_dotenv
import os

# Set page tab display
st.set_page_config(
   page_title="Incremental principal component analysis",
   page_icon= '🖼️',
   layout="wide",
   initial_sidebar_state='collapsed'
)

load_dotenv()

url = os.getenv('LOCAL_API_URL') if str.strip(os.getenv('DEPLOY')) == 'LOCAL' else os.getenv('DOCKER_API_URL')

# App title and description
st.markdown('# Analisando e recriando foto de candidatos eleitos no Brasil 📸')
st.markdown('## Incremental principal component analysis')
st.markdown("---")

n_components = st.slider('Select the number of PCA components.', 0, 100, 1)

cols = st.columns(4)

for idx, col in enumerate(cols):
    with col:
        st.spinner("Wait for it...")
        ### Get bytes from the file buffer

        ### Make request to  API (stream=True to stream response as bytes)
        elected = idx&1 == 1
        bw = idx&2==2

        res = requests.post(url + "/reconstruct_pca_main_components", params={'elected': elected, 'bw': bw, 'n_components': n_components})

        if res.status_code == 200:
            ### Display the image returned by the API
            st.image(res.content, caption=f"This is you {'b&w-' if bw else ''}looking {'an elected' if elected else 'a not elected'} politician in Brazil")#☝️
        else:
            st.markdown("**Oops**, something went wrong 😓 Please try again.")
            print(res.status_code, res.content)










### Create a native Streamlit file upload input
img_camera_buffer = st.camera_input("Let's do a simple face recognition 👇")

if img_camera_buffer is not None:

    cols = st.columns(4)

    for idx, col in enumerate(cols):
        with col:
            st.spinner("Wait for it...")
            ### Get bytes from the file buffer
            img_bytes = img_camera_buffer.getvalue()

            ### Make request to  API (stream=True to stream response as bytes)
            elected = idx&1 == 1
            bw = idx&2==2

            res = requests.post(url + "/reconstruct_pca", params={'elected': elected, 'bw': bw}, files={'img': img_bytes})

            if res.status_code == 200:
                ### Display the image returned by the API
                st.image(res.content, caption=f"This is you {'b&w-' if bw else ''}looking {'an elected' if elected else 'a not elected'} politician in Brazil")#☝️
            else:
                st.markdown("**Oops**, something went wrong 😓 Please try again.")
                print(res.status_code, res.content)

    st.write("""<style>[data-testid="stHorizontalBlock"]{text-align:center;align-items:center;}</style>""",unsafe_allow_html=True)
