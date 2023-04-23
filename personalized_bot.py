import os, config
import gradio as gr
import requests
os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from llama_index import (
    GPTSimpleVectorIndex,
    SimpleDirectoryReader,
    LLMPredictor, )
from langchain.llms import OpenAIChat
import openai
from models import DavinciModel, BaseGPT3Model


openai.api_key = config.OPENAI_API_KEY
elevenLabsAPI = config.ELEVEN_LABS_API_KEY

# If on windows, download and add ffmpeg to path variable
# os.environ['PATH'] += ";C:\\ffmpeg\\bin"

"""
ask_persona

This function is the initial prompting to the user. The user is prompted
with entering one of the preset personas or choosing a custom persona.

Contributors: Shivam Vohra, Jasiu Latocha, Alvin Eizner

@:return persona inputted by user
"""
def ask_persona():
    print("Welcome to the face-to-face chat bot with audio and persona management.")
    print("As a first step, please input the persona that you want to manage.")
    print(
        "There are preset persona's listed below. To use them, enter the number they correspond to. Otherwise, enter a custom persona")
    print("1. Professor Douglas H. Fisher (Vanderbilt University)")
    print("2. Kevin Hart")

    persona_input = (input("\nEnter the persona you want the bot to take on: "))

    return persona_input

"""
initialize

This function initializes the eleven labs voices in order to have a way to map eventual
user input into the requested voice ids.

Contributors: Shivam Vohra, Jasiu Latocha, Alvin Eizner

@:return ids of the voice_id in the index provided
@:return mapping of the voice_id to the name to display
"""
def initialize():
    voices = requests.get("https://api.elevenlabs.io/v1/voices", headers={"xi-api-key": elevenLabsAPI})
    voices = voices.json()['voices']

    ids = {}
    mappings = {}
    for i in range(len(voices)):
        voice = voices[i]
        mappings[i] = voice["name"]
        ids[i] = voice["voice_id"]

    return ids, mappings

"""
generate_ids

This function generates the return voice id and the picture path 
to eventually disply in the gradio UI.

@:param The persona that the user provided to check if it is preset or not

@:return The voice_id to use from eleven labs
@:return The picture path to display in the UI

Contributors: Shivam Vohra, Alvin Eizner, Jasiu Latocha
"""
def generate_ids(persona_input):
    ids, mappings = initialize()

    voice_id = None
    pic_path = None

    if persona_input == '1':
        voice_id = config.ADVISOR_VOICE_ID
        pic_path = config.ADVISOR_IMAGE
    elif persona_input == '2':
        voice_id = config.KEVIN_HART_VOICE_ID
        pic_path = config.KEVIN_HART_IMAGE
    else:
        print("\nIt appears you inputted your own persona.\n")
        print(
            "The voices below are the premade ElevenLabs voices.\nChoose the voice you would like to use by the number it correspond to")

        for num in mappings:
            print(f"{num + 1}. {mappings[num]}")

        voice_input = int(input("Select a voice: "))

        while voice_input <= 0 or voice_input > len(mappings.keys()):
            voice_input = int(input("Improper choice, please reselect: "))

        voice_id = ids[voice_input - 1]
        pic_path = config.DEFAULT_IMAGE

    return voice_id, pic_path

"""
generate_doug

Uses the provided documents about Professor Douglas Fisher to gpt-3.5-turbo for fine-tuning
allowing for the chat bot to act exactly as Douglas Fisher.

@:return The index to query for responses
@:return The LLM predictor

Contributors: Alvin Eizner, Shivam Vohra, Jasiu Latocha
"""
def generate_doug():
    llm = OpenAIChat(temperature=0.5, model_name="gpt-3.5-turbo",
                     prefix_messages=[
                         {"role": "system",
                          "content": f"You are a clone of Vanderbilt University Computer Science Professor Douglas H. Fisher. Answer all questions in the first person, and do mention the fact that you are a clone. There is no need to mention that the provided context is not useful in answering the question, just answer the question. If you do not know the answer to a question based on the context provided, make something up that sounds similar to the data you are trained on. Make sure to match your tone and style of writing to the data you are trained on. Keep your response under 80 words."}
                     ]
                     )
    base_embeddings = OpenAIEmbeddings()
    llm_predictor = LLMPredictor(llm=llm)

    documents = SimpleDirectoryReader('data').load_data()

    index = GPTSimpleVectorIndex.load_from_disk('data.json', llm_predictor=llm_predictor, )

    return index, llm_predictor

"""
choose_model

This function allows the user to choose the model to use whether it be the
Davinci model or the GPT model.

@:param Persona input the user provided
@:return The model to use

Contributors: Shivam Vohra/Alvin Eizner, Jasiu Latocha
"""
def choose_model(persona):

    if persona == "1":
        print("Douglas Fisher has a prebuilt model so we will be using that.")
        return generate_doug()
    else:
        print("Choose to use either GPT3 (1) or the text-davinci-003 model (2).")
        model_num = int(input("Choose model by number from above: "))
        while model_num != 1 and model_num != 2:
            model_num = int(input("Improper model number, choose again: "))

        model = None

        if model_num == 1:
            print("Using Davinci")
            if persona == "2":
                model = DavinciModel("Kevin Hart")
            else:
                model = DavinciModel(persona)
        else:
            print("Using GPT3-Turbo")
            if persona == "2":
                model = BaseGPT3Model("Kevin Hart")
            else:
                model = BaseGPT3Model(persona)

        return model

persona_input = ask_persona()
voice_id, pic_path = generate_ids(persona_input)
model = choose_model(persona_input)

"""
transcribe

This function takes in the audio from gradio and creates the transcript including the input
transcript and the eventual response transcript. The model is queried and then the output
is the audio provided back.

@:param audio that is to be transcribed
@:return Output transcript
@:return File with the output

Contributors: Alvin Eizner, Shivam Vohra, Jasiu Latocha
"""
def transcribe(audio):
    global index, llm_predictor

    # API now requires an extension so we will rename the file
    audio_filename_with_extension = audio + '.wav'
    os.rename(audio, audio_filename_with_extension)

    audio_file = open(audio_filename_with_extension, "rb")
    raw_transcript = openai.Audio.transcribe("whisper-1", audio_file)
    transcript = raw_transcript["text"]
    print("TRANSCRIPT: ", transcript)


    response = None
    if persona_input == "1":
        result = index.query(f"Do not mention the fact that you are a clone, and answer this question: {transcript}?")
        response = str(result)

        phrase = "provided context"
        if phrase in response:
            response = response.split(phrase)[-1].split('. ', 1)[-1]

        phrase = "The original answer remains the same: "
        if phrase in response:
            response = response.replace(phrase, "")

        split_list = response.split(":", 1)
        if len(split_list) > 1:
            response = split_list[1]
    else:
        print(transcript)
        response = model.generate_response(transcript)

    print(f"Bot response: {response}")
    # create a transcript for the app
    chat_transcript = ""
    chat_transcript += f"You: {transcript}\n"
    chat_transcript += f"{persona_input}: {response}\n"

    # text to speech request with eleven labs
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
    data = {
        "text": response.replace('"', ''),
        "voice_settings": {
            "stability": 0.2,
            "similarity_boost": 0.95
        }
    }

    # make a post request to 11Labs to produce audio
    r = requests.post(url, headers={'xi-api-key': config.ELEVEN_LABS_API_KEY}, json=data)

    output_filename = "reply.mp3"
    with open(output_filename, "wb") as output:
        output.write(r.content)

    # return chat_transcript
    return chat_transcript, output_filename


# set a custom theme - Set by Alvin Eizner
theme = gr.themes.Default().set(
    body_background_fill="#000000",
)


# This code launches the gradio UI
# Created by Alvin Eizner
with gr.Blocks(theme=theme) as ui:
    # advisor image input and microphone input
    advisor = gr.Image(value=config.ADVISOR_IMAGE).style(width=config.ADVISOR_IMAGE_WIDTH,
                                                         height=config.ADVISOR_IMAGE_HEIGHT)
    audio_input = gr.Audio(source="microphone", type="filepath")

    # text transcript output and audio
    text_output = gr.Textbox(label="Conversation Transcript")
    audio_output = gr.Audio()

    btn = gr.Button("Run")
    btn.click(fn=transcribe, inputs=audio_input, outputs=[text_output, audio_output])

ui.launch(debug=True, share=True)