import os
import io, base64
import time
from itertools import zip_longest

SYSTEM_PROMPT = "Please reason step by step and put the answer in the \\box{}."

class ChatAPIBot:
    def __init__(self, model_name="gpt-4o-mini", base_url=None):
        self.model_name = model_name
        self.base_url = base_url
        self.custom_api = False
        if("deepseek" in model_name.lower()):
            from openai import OpenAI
            self.client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
            self.chat_function = self.chat_with_openai
        elif("gpt" in model_name.lower()):
            from openai import OpenAI
            self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            self.chat_function = self.chat_with_openai
        elif("doubao" in model_name.lower()):
            from volcenginesdkarkruntime import Ark
            self.client = Ark(
                api_key=os.environ.get("ARK_API_KEY"), 
                timeout=1800,
            )
            self.chat_function = self.chat_with_doubao
        else: # default to openai with custom base url
            if base_url is None:
                raise ValueError("base_url is required for custom openai api model")
            self.custom_api = True
            from openai import OpenAI
            self.client = OpenAI(api_key=os.environ.get("CUSTOM_OPENAI_API_KEY"), base_url=base_url)
            self.chat_function = self.chat_with_custom_openai

    def chat(self, input):
        if self.custom_api or self.base_url is not None:
            return self.chat_function(input, self.model_name, self.base_url)
        return self.chat_function(input, self.model_name)

    def pil_to_base64_url(self, img, format="PNG"):
        buf = io.BytesIO()
        img.save(buf, format=format)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/{format.lower()};base64,{b64}"

    def construct_message(self, texts, images):
        system_message = {"role": "system", "content": SYSTEM_PROMPT}
        user_content = []
        for text, img in zip_longest(texts, images, fillvalue=None):
            if text:
                user_content.append({
                    "type": "text",
                    "text": text
                })
            if img is not None:
                image_url = self.pil_to_base64_url(img)
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                })
        user_message = {"role": "user", "content": user_content}
        return [system_message, user_message]

    def chat_with_openai(self, input, model_name="gpt-4o-mini"):
        texts = input.get("texts", [])
        images = input.get("images", [])
        messages = self.construct_message(texts, images)

        while True:
            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    stream=False
                )
                break
            except Exception as e:
                print(f"Error occurred: {e}. Retrying in 5 seconds...")
                time.sleep(3)
        response = response.choices[0].message.content
        return response

    def chat_with_custom_openai(self, input, model_name="gemini", base_url=""):
        texts = input.get("texts", [])
        images = input.get("images", [])
        messages = self.construct_message(texts, images)

        while True:
            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    stream=False
                )
                break
            except Exception as e:
                print(f"Error occurred: {e}. Retrying in 5 seconds...")
                time.sleep(3)
        response = response.choices[0].message.content
        return response

    def chat_with_doubao(self, input, model_name="doubao-1-5-thinking-pro-m-250428"):
        texts = input.get("texts", [])
        images = input.get("images", [])
        messages = self.construct_message(texts, images)

        print("Sending request to Doubao API...")
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    thinking={
                        "type": "auto",
                    },
                )
                break
            except Exception as e:
                print(f"Error occurred: {e}. Retrying in 5 seconds...")
                time.sleep(3)

        response_finalc = response.choices[0].message.content
        reasoning_content = response.choices[0].message.reasoning_content
        response = "<think>" + reasoning_content + "</think>\n" + response_finalc
        # print(response)
        return response