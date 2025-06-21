import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

def check_auth():
    # Initialize the session state
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    # If already authenticated, skip login
    if st.session_state.authenticated:
        return True

    # Render login UI
    st.title("🔐 Login Required")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if (
            username == os.getenv("APP_USERNAME")
            and password == os.getenv("APP_PASSWORD")
        ):
            st.session_state.authenticated = True
            st.success("✅ Login successful. Please interact again.")
            # Try rerun if available
            try:
                st.experimental_rerun()
            except AttributeError:
                pass  # fallback for older versions
        else:
            st.error("❌ Invalid credentials")

    return False
