import os, config
os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY
from langchain.embeddings import OpenAIEmbeddings
from llama_index import (
    GPTSimpleVectorIndex,
    SimpleDirectoryReader,
    LLMPredictor,)
from langchain.llms import OpenAIChat
import openai

# prepare Q&A embeddings dataframe

openai.api_key = config.OPENAI_API_KEY


class DavinciModel():

    def __init__(self, persona):

        self.persona = persona
        self.persona_prompt = None

        self.generate_persona_prompt()

    def generate_persona_prompt(self):
        prompt = f"Generate a prompt to provide to an LLM about {self.persona}. The prompt should only cover characteristics and information about the persona, but should not include any additional information or scenarios. The output should be a statement about the persona that can then be inputted to another language model to act as it. Only include the prompt in the output with no other text before the prompt and keep the prompt to less than 50 words. Only provide the output prompt, not the prompt given before. The output should also be telling that model that they are the persona."

        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            max_tokens=50,
            temperature=0,
            messages=[{
                "role": "user", "content": prompt
            }]
        )

        generated = chat.choices[0].message.content
        self.persona_prompt = generated



    def generate_response(self, input):

        full_prompt = self.persona_prompt + ". Give an answer only to the question provided based on the previous provided persona, keeping the response to at most 50 words. You are the persona, so respond as though you are the persona, but do not mention you are a clone." + input
        model = openai.Completion.create(
            model="text-davinci-003",
            prompt=full_prompt,
            max_tokens=50,
            temperature=0.7,
            n=1,
            stop=None
        )

        response = model.choices[0].text.strip()

        return response


class BaseGPT3Model():

    def __init__(self, persona):

        self.persona = persona

    def generate_response(self, input):

        model = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            max_tokens=50,
            temperature=0.7,
            messages=[{
                "role": "system", "content": f"You are a clone of {self.persona}. You will be asked questions and must respond in the manner that the persona provided would. Do not mention that you are a clone and respond with at most 50 words."
            }, {
                "role": "user", "content": input
            }]
        )
        response = model.choices[0].message.content
        return response
