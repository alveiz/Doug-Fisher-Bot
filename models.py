import os, config
os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY
import openai

# prepare Q&A embeddings dataframe

openai.api_key = config.OPENAI_API_KEY


class DavinciModel():

    def __init__(self, persona):

        self.persona = persona
        self.persona_prompt = None

        self.generate_persona_prompt()

    def generate_persona_prompt(self):
        prompt = "Generate a specific prompt to provide to an LLM that is based off of the persona of " + self.persona + ". Please only provide the persona prompt and no additional text or information"

        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            max_tokens=35,
            messages=[{
                "role": "user", "content": prompt
            }]

        )

        generated = chat.choices[0].message.content
        self.persona_prompt = generated


    def generate_response(self, input):

        full_prompt = self.persona_prompt + " " + input
        model = openai.Completion.create(
            model="text-davinci-003",
            prompt=full_prompt,
            max_tokens=100,
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
            max_tokens=100,
            temperature=0.7,
            messages=[{
                "role": "system", "content": f"Take on the personality of {self.persona}. Respond to questions exactly how {self.persona} would respond."
            }, {
                "role": "user", "content": input
            }]
        )
        response = model.choices[0].message.content
        return response





