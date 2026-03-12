import sys
from pathlib import Path

# Add scripts directory to sys.path to import config.py
project_root = Path(__file__).resolve().parent.parent.parent.parent
scripts_dir = project_root / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.append(str(scripts_dir))
import config

from openai import OpenAI


class LLM:
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = model
    
    def generate(self, prompt: str) -> str:
        """Generate a response from the LLM."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
