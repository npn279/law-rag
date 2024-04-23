import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import dotenv
dotenv.load_dotenv()

import logging
from openai import OpenAI


OPENAI_CHAT_MODEL_NAME = os.getenv("OPENAI_CHAT_MODEL_NAME")


class OPENAI:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def get_response(self, prompt, **kwargs):
        try:
            temperature = kwargs.get('temperature', 0)
            max_tokens = kwargs.get('max_tokens', 4000)
            stream = kwargs.get('stream', False)
            history = kwargs.get('history', [])
            system_prompt = kwargs.get('system_prompt', "You are a helpful assistant.")
            model = kwargs.get('model', OPENAI_CHAT_MODEL_NAME)

            completion = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    *history,
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
            )

            return completion
        except Exception as e:
            logging.error(e)
            return None