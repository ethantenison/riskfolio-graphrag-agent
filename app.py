"""Hugging Face Spaces entry point."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from riskfolio_graphrag_agent.app.gradio_ui import create_gradio_app

demo = create_gradio_app()

demo.launch(ssr_mode=False)
