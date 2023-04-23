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

This function asks the user to input a persona. This can be a custom
persona or it can be the preset personas of Kevin Hart and Professor Douglas Fisher

@:return Persona that the user inputted

Contributors: Shivam Vohra/Jasiu Latocha, Alvin Eizner
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

This function initializes the mappings of the names and ids from the
eleven labs API.

@:return Mapping of the index to ids
@:return Mapping of the index to voice names

Contributors: Shivam Vohra, Alvin Eizner, Jasiu Latocha
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

This function picks the voice id to use from the user's input as well as
the picture path to display

@:param Persona that the user inputted
@:return ElevenLabs voice id
@:return Picture path to display in the UI

Contributors: Alvin Eizner/Shivam Vohra, Jasiu Latocha
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
        # print("You chose kevin hart!")
        # print("pic path: ", pic_path)
        # print("voice id: ", voice_id)
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
        try:
            dalle_image = openai.Image.create(
                prompt=f"Create a photorealistic image of {persona_input}",
                n=1,
                size="1024x1024")
            pic_path = dalle_image["data"][0]["url"]
        except openai.error.InvalidRequestError:
            print("Using default image")

    return voice_id, pic_path

# This snippet of code generates the Douglas Fisher model to use - Alvin Eizner
llm = OpenAIChat(temperature=0.5, model_name="gpt-3.5-turbo",
                 prefix_messages=[
                     {"role": "system",
                      "content": f"You are a clone of Vanderbilt University Computer Science Professor Douglas H. Fisher. Answer all questions in the first person, and do mention the fact that you are a clone. There is no need to mention that the provided context is not useful in answering the question, just answer the question. If you do not know the answer to a question based on the context provided, make something up that sounds similar to the data you are trained on. Make sure to match your tone and style of writing to the data you are trained on. Keep your response under 80 words."}])
llm_predictor = LLMPredictor(llm=llm)


"""
choose_model

This function chooses the model from the persona input provided, selecting either
the pretrained Douglas Fisher model or a Davinci or GPT based model.

@:param Persona provided by user
@:return Model selected

Contributors: Shivam Vohra/Alvin Eizner, Jasiu Latocha
"""
def choose_model(persona):
    if persona == "1":
        print("Douglas Fisher has a prebuilt model so we will be using that.")
    else:
        print("Choose to use either GPT3 (1) or the text-davinci-003 model (2).")
        model_num = int(input("Choose model by number from above: "))
        while model_num != 1 and model_num != 2:
            model_num = int(input("Improper model number, choose again: "))

        model = None

        if model_num == 1:
            print("Using GPT3 Turbo")
            if persona == "2":
                model = BaseGPT3Model("Kevin Hart")
            else:
                model = BaseGPT3Model(persona)
        else:
            print("Using Davinci")
            if persona == "2":
                model = DavinciModel("Kevin Hart")
            else:
                model = DavinciModel(persona)

        return model


persona_input = ask_persona()
voice_id, pic_path = generate_ids(persona_input)
model = choose_model(persona_input)

"""
transcribe

Converts the audio to transcripts that are also provided to gradio
for eventual output audio

@:param The audio from gradio
@:return The output transcript
@:return Output file of reply

Contributors: Alvin Eizner, Shivam Vohra, Jasiu Latocha
"""
def transcribe(audio):
    global llm_predictor

    # API now requires an extension so we will rename the file
    audio_filename_with_extension = audio + '.wav'
    os.rename(audio, audio_filename_with_extension)

    audio_file = open(audio_filename_with_extension, "rb")
    raw_transcript = openai.Audio.transcribe("whisper-1", audio_file)
    transcript = raw_transcript["text"]
    print("TRANSCRIPT: ", transcript)

    response = None
    if persona_input == "1":
        index = GPTSimpleVectorIndex.load_from_disk('data.json', llm_predictor=llm_predictor, )
        result = index.query(f"Do not mention the fact that you are a clone, and answer this question: {transcript}")
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
        response = model.generate_response(
            f"Do not mention the fact that you are a clone, and answer this question: {transcript}")

    print(f"Bot response: {response}")
    # create a transcript for the app
    chat_transcript = ""
    chat_transcript += f"You: {transcript}\n"
    chat_transcript += f"{persona_input}: {response}\n"

    # text to speech request with eleven labs
    print("out", voice_id)
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


# set a custom theme
theme = gr.themes.Default().set(
    body_background_fill="#000000",
)


# This launches the gradio UI - Alvin Eizner, Shivam Vohra
with gr.Blocks(theme=theme) as ui:
    # advisor image input and microphone input

    advisor = gr.Image(value=pic_path).style(width=config.ADVISOR_IMAGE_WIDTH,
                                             height=config.ADVISOR_IMAGE_HEIGHT)
    audio_input = gr.Audio(source="microphone", type="filepath")

    # text transcript output and audio
    text_output = gr.Textbox(label="Conversation Transcript")
    audio_output = gr.Audio()

    btn = gr.Button("Run")
    btn.click(fn=transcribe, inputs=audio_input, outputs=[text_output, audio_output])

ui.launch(debug=True, share=True)