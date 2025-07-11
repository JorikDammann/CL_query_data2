import os
from langchain_community.vectorstores import Chroma  #vector search DB backend
from langchain_community.embeddings import OpenAIEmbeddings  #OpenAI text embedding model
from langchain.text_splitter import CharacterTextSplitter  #breaks up long texts for better search
from langchain_community.document_loaders import TextLoader  #generic text file loader
from langchain_community.document_loaders import JSONLoader  #handles structured JSON loading



OPENAI_API_KEY = ""
if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

#Load datapoints from *_datapoints.json files
def load_all_json_docs(folder_path="./output_visuals"):
    docs = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            #only pull datapoints (e.g., steel price facts, EAF capacity, etc.)
            if file.endswith(".json") and "_datapoints" in file:
                full_path = os.path.join(root, file)
                #pull only the "datapoint_text" from each object in the JSON array
                loader = JSONLoader(file_path=full_path, jq_schema=".[] | .datapoint_text", text_content=False) # JQ pulls datapoint_text field from every item
                docs.extend(loader.load()) #load and append each datapoint as a document
    return docs


#Embed and save into Chroma vectorstore
docs = load_all_json_docs()
print(f"Loaded {len(docs)} documents.")

#break long datapoints into 500-char chunks with 50-char overlap
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(docs)

embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

#embed and store everything in a Chroma vector DB
db = Chroma.from_documents(split_docs, embedding, persist_directory="./tea_vectorstore")
db.persist()  # ake the DB persistent on disk

print("Vectorstore built!")
