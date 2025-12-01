"""
Crop Disease Diagnosis System - Main Application Entry Point

This is the main Streamlit application that provides a web interface for
crop disease diagnosis using a CNN model.
"""

import streamlit as st


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Crop Disease Diagnosis System",
        page_icon="🌾",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🌾 Crop Disease Diagnosis System")
    st.write("AI-powered disease detection for rice and pulse crops")
    
    # Placeholder for future implementation
    st.info("Application setup complete. Components will be integrated in subsequent tasks.")


if __name__ == "__main__":
    main()
