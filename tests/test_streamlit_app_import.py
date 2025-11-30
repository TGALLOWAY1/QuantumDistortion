def test_streamlit_app_imports() -> None:
    # Importing the app module should not raise exceptions
    import quantum_distortion.ui.app_streamlit as app  # noqa: F401

