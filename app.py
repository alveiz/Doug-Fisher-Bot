import os, config, requests
import gradio as gr
os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from llama_index import (
    GPTSimpleVectorIndex, 
    SimpleDirectoryReader, 
    LLMPredictor,)
from langchain.llms import OpenAIChat
import openai
openai.api_key = config.OPENAI_API_KEY

messages = [{"role": "system", "content": 'You are a financial advisor. Respond to all input in 50 words or less. Answer in the first person. Do not use the $ sign, write out dollar amounts with the full word dollars. Do not use quotation marks. Do not say you are an AI language model.'}]

# prepare Q&A embeddings dataframe

llm=OpenAIChat(temperature=0.5, model_name="gpt-3.5-turbo", 
               prefix_messages=[
    {"role": "system", "content": f"You are a clone of Vanderbilt University Computer Science Professor Douglas H. Fisher. Answer all questions in the first person, and do mention the fact that you are a clone. There is no need to mention that the provided context is not useful in answering the question, just answer the question. If you do not know the answer to a question based on the context provided, make something up that sounds similar to the data you are trained on. Make sure to match your tone and style of writing to the data you are trained on. Keep your response under 80 words."}
    ]
)
base_embeddings = OpenAIEmbeddings()
llm_predictor = LLMPredictor(llm=llm)

documents = SimpleDirectoryReader('data').load_data()

index = GPTSimpleVectorIndex.load_from_disk('data.json', llm_predictor=llm_predictor,)

def transcribe(audio):
    global index, llm_predictor

    # API now requires an extension so we will rename the file
    audio_filename_with_extension = audio + '.wav'
    os.rename(audio, audio_filename_with_extension)

    audio_file = open(audio_filename_with_extension, "rb")
    raw_transcript = openai.Audio.transcribe("whisper-1", audio_file)
    transcript = raw_transcript["text"]
    print("TRANSCRIPT: ", transcript)

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

    chat_transcript = ""
    chat_transcript += f"Student: {transcript}\n"
    chat_transcript += f"Professor Fisher: {response}\n"

    # text to speech request with eleven labs
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{config.ADVISOR_VOICE_ID}/stream"
    data = {
        "text": response.replace('"', ''),
        "voice_settings": {
            "stability": 0.2,
            "similarity_boost": 0.95
        }
    }

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

with gr.Blocks(theme=theme) as ui:
    # advisor image input and microphone input
    advisor = gr.Image(value=config.ADVISOR_IMAGE).style(width=config.ADVISOR_IMAGE_WIDTH, height=config.ADVISOR_IMAGE_HEIGHT)
    audio_input = gr.Audio(source="microphone", type="filepath")

    # text transcript output and audio 
    text_output = gr.Textbox(label="Conversation Transcript")
    audio_output = gr.Audio()

    btn = gr.Button("Run")
    btn.click(fn=transcribe, inputs=audio_input, outputs=[text_output, audio_output])

ui.launch(debug=True, share=True)