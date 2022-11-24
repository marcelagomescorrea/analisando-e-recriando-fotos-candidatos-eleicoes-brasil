import streamlit as st
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

if st.secrets['URL_API']:
    url = st.secrets['URL_API']
    st.markdown(url)

# App title and description
st.markdown('# Analisando e recriando foto de candidatos eleitos no Brasil 📸')
st.markdown('## Incremental principal component analysis')
st.markdown("---")

n_components = st.slider('Select the number of PCA components.', 1, 100, 1)

upper_cols = st.columns(4)

for idx, col in enumerate(upper_cols):
    with col:
        st.spinner("Wait for it...")

        ### Make request to  API (stream=True to stream response as bytes)
        elected = idx&1 == 1
        bw = idx&2==2

        res = requests.post(url + "/reconstruct_random", params={'model': 'pca', 'elected': elected, 'bw': bw, 'n_components': n_components})

        if res.status_code == 200:
            ### Display the image returned by the API
            st.image(res.content, caption=f"This is main-components {'b&w-' if bw else ''}looking {'an elected' if elected else 'a not elected'} politician in Brazil ☝️")#☝️
        else:
            st.markdown("**Oops**, something went wrong 😓 Please try again.")
            print(res.status_code, res.content)

### Create a native Streamlit file upload input
img_camera_buffer = st.camera_input("Let's do a simple face recognition 👇")

if img_camera_buffer is not None:

    lower_cols = st.columns(4)

    for idx, col in enumerate(lower_cols):
        with col:
            st.spinner("Wait for it...")
            ### Get bytes from the file buffer
            img_bytes = img_camera_buffer.getvalue()

            ### Make request to  API (stream=True to stream response as bytes)
            elected = idx&1 == 1
            bw = idx&2==2

            res = requests.post(url + "/reconstruct_photo", params={'model': 'pca', 'elected': elected, 'bw': bw}, files={'img': img_bytes})

            if res.status_code == 200:
                ### Display the image returned by the API
                st.image(res.content, caption=f"This is you {'b&w-' if bw else ''}looking {'an elected' if elected else 'a not elected'} politician in Brazil ☝️")
            else:
                st.markdown("**Oops**, something went wrong 😓 Please try again.")
                print(res.status_code, res.content)

    st.write("""<style>[data-testid="stHorizontalBlock"]{text-align:center;align-items:center;}</style>""",unsafe_allow_html=True)
