# Doug-Fisher-Bot
A virtual clone of Professor Doug Fisher

## App Description
This app works by using a combination of the langchain python library (https://python.langchain.com/en/latest/index.html) and the llama index python library (https://gpt-index.readthedocs.io/en/latest/index.html).
Specificially this relies on langchain and lamma index to create an LLM and LLM Predicor with OpenAI gpt-3.5-turbo model. The app then uses the GPTSimpleVectorIndex from llama index to index all the data from the LLM.
The index is stored n the data.json file.

## setup
clone this repo and run
`
pip install -r requirements.txt
`
Then create a config.py file in the main directory. It should look something like this:  
`
OPENAI_API_KEY = "sk-..."  
ELEVEN_LABS_API_KEY = "11Labs API KEY"  

ADVISOR_IMAGE_WIDTH = 360  
ADVISOR_IMAGE_HEIGHT = 360  

ADVISOR_IMAGE = "images/doug.png"  

ADVISOR_VOICE_ID = "11Labs VOICE ID"
`

## running the app
`
python3 app.py
`