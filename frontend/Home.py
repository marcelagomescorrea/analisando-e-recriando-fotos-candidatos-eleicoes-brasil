import streamlit as st
import requests

# Set page tab display
st.set_page_config(
   page_title="Analisando e recriando foto de candidatos eleitos no Brasil",
   page_icon= '🖼️',
   layout="wide",
   initial_sidebar_state="expanded",
)

url = None
if st.secrets['DEPLOY'] == 'LOCAL':
    url = st.secrets['LOCAL_API_URL']
elif st.secrets['DEPLOY'] == 'DOCKER':
    url = st.secrets['DOCKER_API_URL']
else:
    url = st.secrets['CLOUD_API_URL']

# App title and description
st.markdown('# Analisando e recriando foto de candidatos eleitos no Brasil 📸')
st.markdown('### Disclaimer: this is a data science project designed for academic purposes. Do not take it seriously. ')

st.markdown("---")

st.markdown('### Incremental Principal Component Analysis Reconstruction')

st.markdown('#### Mean')

high_upper_cols = st.columns(4)

for idx, col in enumerate(high_upper_cols):
    with col:
        st.spinner("Wait for it...")

        ### Make request to  API (stream=True to stream response as bytes)
        elected = idx&1 == 1
        bw = idx&2==2

        res = requests.post(url + "/reconstruct_pca_mean", params={'elected': elected, 'bw': bw})

        if res.status_code == 200:
            ### Display the image returned by the API
            st.image(res.content, caption=f"This is mean-components {'b&w-' if bw else ''}looking {'an elected' if elected else 'a not elected'} politician in Brazil ☝️")#☝️
        else:
            st.markdown("**Oops**, something went wrong 😓 Please try again.")
            print(res.status_code, res.content)

st.markdown('#### Main components')

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

st.markdown("---")

st.markdown('### Variational Autoencoder Random Reconstruction')

lower_cols = st.columns(4)

for idx, col in enumerate(lower_cols):
    with col:
        st.spinner("Wait for it...")

        ### Make request to  API (stream=True to stream response as bytes)
        elected = idx&1 == 1
        bw = idx&2==2

        res = requests.post(url + "/reconstruct_random", params={'model': 'autoencoder', 'elected': elected, 'bw': bw, 'n_components': 0})

        if res.status_code == 200:
            ### Display the image returned by the API
            st.image(res.content, caption=f"This is a randomly-generated {'b&w-looking' if bw else ''} {'elected' if elected else 'not elected'} politician in Brazil ☝️")#☝️
        else:
            st.markdown("**Oops**, something went wrong 😓 Please try again.")
            print(res.status_code, res.content)
