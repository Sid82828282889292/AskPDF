import streamlit.components.v1 as components
import urllib.parse
import os

def render_pdf_viewer(pdf_file_name: str, highlight_text: str = None):
    """Render the PDF with optional highlight text."""
    viewer_path = os.path.join("static", "pdfjs", "viewer.html")
    encoded_file = urllib.parse.quote(pdf_file_name)

    # Prepare highlight query
    highlight_query = ""
    if highlight_text:
        encoded_highlight = urllib.parse.quote(highlight_text)
        highlight_query = f"&highlight={encoded_highlight}"

    # Final viewer URL
    viewer_url = f"{viewer_path}?file={encoded_file}{highlight_query}"

    components.iframe(viewer_url, height=700, scrolling=True)
