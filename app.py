"""Hugging Face Spaces entry point."""

from riskfolio_graphrag_agent.app.gradio_ui import create_gradio_app

demo = create_gradio_app()

if __name__ == "__main__":
    demo.launch()
