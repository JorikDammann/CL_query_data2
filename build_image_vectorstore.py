import os
import json
from langchain_community.vectorstores import Chroma  #vector DB backend
from langchain_community.embeddings import OpenAIEmbeddings  #to embed figure descriptions
from langchain_core.documents import Document  #wraps text + metadata
from dotenv import load_dotenv  #allows environment variables from .env

load_dotenv()
OPENAI_API_KEY = ""

#Set up embedding and vectorstore path
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
persist_dir = "./image_vectorstore"

#Walk through all figure JSON files in ./output_visuals
docs = []
root = "./output_visuals"

for dirpath, _, filenames in os.walk(root):
    for file in filenames:

        #Only use visual figure metadata (not datapoint-only JSONs)
        if file.endswith(".json") and not file.endswith("_datapoints.json"):
            with open(os.path.join(dirpath, file)) as f:
                data = json.load(f)
                content = data.get("description", "").strip()

                #Skip any figures with no meaningful description
                if not content:
                    continue

                metadata = {
                    "caption": data.get("caption", ""),
                    "filename": data.get("filename", ""),
                    "source_link": data.get("source_link", ""),
                    "image_path": os.path.join(dirpath, data.get("filename", "")),
                    "tags": ", ".join(data.get("tags", [])) #join tags as comma-sep string
                }

                #Add to documents list for embedding
                docs.append(Document(page_content=content, metadata=metadata))

if not docs:
    raise ValueError("No figure descriptions found to embed. Check your data.") #sanity check

#Embed and store in Chroma
db = Chroma.from_documents(documents=docs, embedding=embedding, persist_directory=persist_dir)
db.persist()
print(f"Indexed {len(docs)} figures with descriptions to ./image_vectorstore")
