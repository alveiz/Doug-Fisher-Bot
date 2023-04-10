import os, config
os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from llama_index import (
    GPTSimpleVectorIndex, 
    SimpleDirectoryReader, 
    LLMPredictor,)
from langchain.llms import OpenAIChat

llm=OpenAIChat(temperature=0.5, model_name="gpt-3.5-turbo", 
               prefix_messages=[
    {"role": "system", "content": f"You are a clone of Vanderbilt University Computer Science Professor Douglas H. Fisher. Answer all questions in the first person, and do mention the fact that you are a clone. There is no need to mention that the provided context is not useful in answering the question, just answer the question. If you do not know the answer to a question based on the context provided, make something up that sounds similar to the data you are trained on. Make sure to match your tone and style of writing to the data you are trained on. Keep your response under 80 words."}
    ]
)
base_embeddings = OpenAIEmbeddings()
llm_predictor = LLMPredictor(llm=llm)

documents = SimpleDirectoryReader('data').load_data()
#index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor,)
#index.save_to_disk('data.json')
index = GPTSimpleVectorIndex.load_from_disk('data.json', llm_predictor=llm_predictor,)
result = index.query("What do you think about artificial intelligence? Do not mention the fact that you are a clone.")

response = str(result)
phrase = "provided context"
if phrase in response:
    print("yes\n")
    response = response.split(phrase)[-1].split('. ', 1)[-1]

phrase = "The original answer remains the same: "
if phrase in response:
    response = response.replace(phrase, "")
print(response)
