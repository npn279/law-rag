import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import dotenv
dotenv.load_dotenv()

import logging
import google.generativeai as genai 


GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME")

class GEMINI:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)

    def generate(self, prompt, **kwargs):
        try:
            stream = kwargs.get("stream", False)
            temperature = kwargs.get("temperature", 0.9)
            top_k = kwargs.get("top_k", 1)
            top_p = kwargs.get("top_p", 1)
            max_output_tokens = kwargs.get("max_output_tokens", 2048)
            model_name = kwargs.get("model_name", GEMINI_MODEL_NAME)

            generation_config = {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_output_tokens": max_output_tokens,
            }

            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE"
                },
            ]

            safety_settings = kwargs.get("safety_settings", safety_settings)

            model = genai.GenerativeModel(model_name=model_name,
                                        generation_config=generation_config,
                                        safety_settings=safety_settings)
            prompt_parts = [prompt]
            response = model.generate_content(prompt_parts, stream=stream)
            return response
        except:
            print("Exception in GEMINI.generate()")
            return None