import sys
import asyncio

if sys.platform.startswith('win') and sys.version_info >= (3, 8):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
from agent import run_agent, agent, AgentDeps
import os
from pathlib import Path

st.title("AI Sight - Computer Vision Assistant")





# Create a file uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Create a directory to save uploaded files if it doesn't exist
    upload_dir = Path("agent/public/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the uploaded file
    file_path = upload_dir / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display the image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Create a button to run analysis
    if st.button("Analyze Image"):
        with st.spinner("Analyzing image..."):
            result = asyncio.run(run_agent())
            
            # Display results
            st.write("Analysis Results:")
            st.write(result)

st.sidebar.markdown("""
## About
This app uses AI to analyze images. You can:
1. Upload an image
2. Get classification results
3. Detect objects in the image
""")
