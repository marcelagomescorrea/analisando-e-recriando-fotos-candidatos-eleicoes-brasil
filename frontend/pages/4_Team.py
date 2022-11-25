import requests
import streamlit as st

# Set page tab display
st.set_page_config(
   page_title="Team",
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
st.markdown('# Team')
st.markdown("---")

avatars = [
    ['Hélio Bomfim De Macêdo Filho', 'https://avatars.githubusercontent.com/u/2234749'],
    ['Marcela Gomes Corrêa', 'https://avatars.githubusercontent.com/u/53438797'],
    ['Marcio Colazingari', 'https://avatars.githubusercontent.com/u/53103613']
]

lower_cols = st.columns(3)

for idx, col in enumerate(lower_cols):
    with col:
        st.spinner("Wait for it...")
        ### Get bytes from the file buffer
        photo = requests.get(avatars[idx][1])
        if photo.status_code != 200:
            st.markdown("**Oops**, something went wrong 😓 Please try again.")
            continue
        photo_bytes = photo.content

        res = requests.post(url + "/reconstruct_photo", params={'model': 'pca', 'elected': False, 'bw': True}, files={'img': photo_bytes})

        if res.status_code == 200:
            ### Display the image returned by the API
            st.image(res.content, caption=avatars[idx][0])
        else:
            st.markdown("**Oops**, something went wrong 😓 Please try again.")
            print(res.status_code, res.content)
