import os
import json
from langchain.schema import Document  #the LangChain Document format for search-ready chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter  #for breaking text into chunks
from langchain.embeddings import OpenAIEmbeddings  #OpenAI embedding model
from langchain.vectorstores import Chroma  #vector DB for storing/retrieving based on similarity


OUTPUT_FOLDER = "./output_visuals"

#helper function to walk through output folders and collect all descriptions + datapoints
def load_documents_from_json(output_dir):
    docs = []
    for root, dirs, files in os.walk(output_dir):  #recursively go through subfolders
        for file in files:

            #pull in OCR figure descriptions (ignore datapoints for now)
            if file.endswith(".json") and "_datapoints" not in file:
                path = os.path.join(root, file)
                with open(path) as f:
                    meta = json.load(f)
                content = meta.get("description", "")
                tags = meta.get("tags", [])
                if content:
                    docs.append(Document(
                        page_content=content,
                        metadata={
                            "source": meta.get("source_link"), #helpful for click-through or UI preview
                            "image": meta.get("filename"),  #match to screenshot
                            "tags": tags #potential use cases later. BRAINSTORM
                        }
                    ))

            #pull in structured datapoints separately (short bullet-style facts)
            elif file.endswith("_datapoints.json"):
                path = os.path.join(root, file)
                with open(path) as f:
                    points = json.load(f)
                for dp in points:
                    docs.append(Document(
                        page_content=dp["datapoint_text"],
                        metadata={"source": dp["source_url"]}
                    ))
    return docs

#grab all figure descriptions and datapoints as LangChain Documents
documents = load_documents_from_json(OUTPUT_FOLDER)

#split longer entries into chunks (300 characters, no overlap)
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)
split_docs = splitter.split_documents(documents)

#embed those chunks and save to a local Chroma vector DB (with OpenAI)
db = Chroma.from_documents(split_docs, OpenAIEmbeddings(), persist_directory="./tea_vectorstore") #this folder will contain the db files

# save the database to disk so it persists between runs
db.persist()
print("Vectorstore built and saved ! Yess")
