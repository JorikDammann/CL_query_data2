import os
import streamlit as st
from PIL import Image
from langchain_community.vectorstores import Chroma  #vector DB
from langchain_community.embeddings import OpenAIEmbeddings  #for embedding text
from langchain.chat_models import ChatOpenAI  #(optional, not used here)
from langchain.chains import RetrievalQA  #(optional, not used here)
from dotenv import load_dotenv  #for loading env vars like API key

load_dotenv()
OPENAI_API_KEY = ""
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not set.")
    st.stop()

#Load pre-built vectorstore of figure descriptions
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
db = Chroma(persist_directory="./image_vectorstore", embedding_function=embedding)

#Set up Streamlit UI - Chat GPT's suggestion
st.set_page_config(page_title="Figure Search (EAF TEA)", layout="wide")
st.title("Search EAF Figures by Description")

#Input box for user search
query = st.text_input("Describe the type of figure you’re looking for (e.g. 'emissions trend by process'):")

#when user submits a search
if query:
    results = db.similarity_search(query, k=5)
    st.write(f"Top {len(results)} matching figures:")

    #display each result
    for r in results:
        meta = r.metadata
        image_path = meta.get("image_path", "")
        if os.path.exists(image_path):
            st.image(image_path, width=600, caption=meta.get("caption", ""))
        st.markdown(f"**Description:** {r.page_content}")
        st.markdown(f"**Tags:** {meta.get('tags', '')}")
        st.markdown(f"[🔗 Source]({meta.get('source_link', '')})")
        st.markdown("---")

#UP NEXT-- Firstly, the search kinda sucks rn. Most of this has to do with the data + descriptions database. IMPROVE structure. 
#I also want to do some simple keyword search of the reports
#Also, rn it only goes through images, not text. That should be an easy addition
# Also, I want to add a split screen feature where if you click one of the results, it pulls up the source document on the right side (or at least a screenshot of the page)
#Also, I'd like to add a sanity check GPT estimate. This is basically like asking the same question to ChatGPT and checking if its same ballpark. 

