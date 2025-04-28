import requests

import streamlit as st

st.title("AI Sight - Computer Vision Assistant")

if st.button("Analyze Image"):
    with st.spinner("Analyzing image..."):
        try:
            res = requests.get("http://127.0.0.1:8000/classify_image")
            if res.status_code == 200:
                result = res.json()
                st.write("Analysis Results:")
                st.write(result)
        except Exception as e:
            st.write(f"Waiting for agent... {e}")


