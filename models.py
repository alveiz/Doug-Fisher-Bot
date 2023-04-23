import os, config
os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY
import openai

# prepare Q&A embeddings dataframe

openai.api_key = config.OPENAI_API_KEY

"""
DavinciModel

OpenAI model that functions on instructions and completing statements
based on those instructions
"""
class DavinciModel():

    """
    __init__

    Constructor for model

    @:param Persona that was provided

    Contributors: Shivam Vohra/Jasiu Latocha, Alvin Eizner
    """
    def __init__(self, persona):

        self.persona = persona
        self.persona_prompt = None

        self.generate_persona_prompt()

    """
    generate_persona_prompt
    
    Generates the persona prompt from ChatGPT that will be inputted into the specific models
    to act as the persona
    
    Contributors: Alvin Eizner/Jasiu Latocha/Shivam Vohra
    """
    def generate_persona_prompt(self):
        prompt = f"Generate a prompt to provide to an LLM about {self.persona}. The prompt should only cover characteristics and information about the persona, but should not include any additional information or scenarios. The output should be a statement about the persona that can then be inputted to another language model to act as it. Only include the prompt in the output with no other text before the prompt. Only provide the output prompt, not the prompt given before. The output should also be telling that model that they are the persona."

        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            max_tokens=60,
            temperature=0,
            messages=[
                {"role": "system", "content": f"You are an expert at creating prompts that give more context about a persona to an LLM."},
                {"role": "user", "content": prompt}
            ]

        )

        generated = chat.choices[0].message.content
        self.persona_prompt = generated


    """
    generate_response
    
    This function uses OpenAI to generate the response from the inputted statement
    
    @:param Input question supplied to respond to
    @:return The response from the model
    
    Contributors: Shivam Vohra, Alvin Eizner, Jasiu Latocha
    """
    def generate_response(self, input):
        print("persona prompt:", self.persona_prompt)

        add = "."
        if self.persona == "Kevin Hart":
            " Make your answer be a bit edgy, and/or contain curse words like 'shit'."

        full_prompt = self.persona_prompt + add + " Give an answer only to the question provided based on the previous provided persona, keeping the response to at most 50 words. You are the persona, so respond as though you are the persona, but do not mention you are a clone.\nQ:" + input + "\nA:"
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


"""
BaseGPT3Model

OpenAI model that functions on prompting and receiving responses
to those prompts
"""
class BaseGPT3Model():
    """
    __init__

    Constructor for model

    @:param Persona that was provided

    Contributors: Jasiu Latocha/Shivam Vohra/Alvin Eizner
    """
    def __init__(self, persona):

        self.persona = persona

    """
    generate_response
    
    This function uses OpenAI to generate the response from the inputted statement
    
    @:param Input question supplied to respond to
    @:return The response from the model
    
    Contributors: Shivam Vohra, Jasiu Latocha/Alvin Eizner
    """
    def generate_response(self, input):
        add = "."
        if self.persona == "Kevin Hart":
            add = " how Kevin Hart would. Make your response be edgy and/or contain curse words like 'shit'."
        model = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            max_tokens=100,
            temperature=0.7,
            messages=[{
                "role": "system", "content": f"Take on the personality of {self.persona}. Respond to questions similar to how {self.persona} would respond" + add
            }, {
                "role": "user", "content": f"Please respond to this question{add}: {input}"
            }]
        )
        response = model.choices[0].message.content
        return response
