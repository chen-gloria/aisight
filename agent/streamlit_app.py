import streamlit as st
from PIL import Image
from agent import geo_guess_location_from_image
import httpx
import asyncio

st.set_page_config(
    page_title="Photo to Geo location guesser",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.header("Guess location from image")

with st.sidebar:
    st.header("Location guessing app")
    st.write(
        "This app uses a generative model (gemini-2.0-flash) to guess the geographical location of the chosen photo."
    )
    st.header("How to use this app")
    st.write("1. Upload a photo from local device (jpg) or from web link (jpg or png)")
    st.write("2. Scroll down and click 'Guess the location' button")
    st.write("3. Wait for a bit, it will show the output below the button")

uploaded_file = st.file_uploader(
    "Upload image file", accept_multiple_files=False, type=["jpg"]
)


def reset():
    st.session_state.clear()


if uploaded_file is None:
    st.write("or")
    image_link = st.text_input("Enter image link")
else:
    image_link = ""

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)
    image_bytes_data = uploaded_file.getvalue()

    generate = st.button("Guess the location", type="primary", use_container_width=True)

    if generate:
        st.write("Let's guess the location now")
        with st.spinner("Generating..."):
            # Create an event loop and run the async function
            result = asyncio.run(geo_guess_location_from_image(image_bytes_data))
            st.write(result.data)
        st.success("Done!")
        st.button("Reset", on_click=reset)

if len(image_link) > 0:
    try:
        # Validate if the link is a valid URL
        if not image_link.startswith(("http://", "https://")):
            st.error("Please enter a valid URL starting with http:// or https://")
        else:
            # Try to fetch the image
            image_response = httpx.get(image_link)
            image_response.raise_for_status()  # Raise an exception for HTTP errors

            # Check if the content type is an image
            content_type = image_response.headers.get("content-type", "")
            if not content_type.startswith("image/"):
                st.error(
                    f"The URL does not point to an image. Content type: {content_type}"
                )
            else:
                image_bytes_data = image_response.content

                # Display the image
                st.image(
                    image_bytes_data, caption="Image from URL", use_container_width=True
                )

                generate = st.button(
                    "Guess the location", type="primary", use_container_width=True
                )

                if generate:
                    st.write("Let's guess the location now")
                    with st.spinner("Generating..."):
                        # Create an event loop and run the async function
                        result = asyncio.run(
                            geo_guess_location_from_image(image_bytes_data)
                        )
                        st.write(result.data)
                    st.success("Done!")

                    st.button("Reset", on_click=reset)

    except httpx.RequestError as e:
        st.error(f"Error fetching the image: {str(e)}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
