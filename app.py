from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

load_dotenv()
import os

from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings

from data import load_db
from names import DATASET_ID, MODEL_ID

@st.cache_resource
def init():
    embeddings = OpenAIEmbeddings(model=MODEL_ID)
    dataset_path = f"hub://{os.environ['ACTIVELOOP_ORG_ID']}/{DATASET_ID}"

    db = load_db(
        dataset_path,
        embedding_function=embeddings,
        token=os.environ["ACTIVELOOP_TOKEN"],
        org_id=os.environ["ACTIVELOOP_ORG_ID"],
        read_only=True,
    )

    prompt = PromptTemplate(
        input_variables=["content"],
        template=Path("prompts/bot.prompt").read_text(),
    )

    llm = ChatOpenAI(temperature=0.7)

    chain = LLMChain(llm=llm, prompt=prompt)

    return db, chain

db, chain = init()

st.title("Disney song for you")

text_input = st.text_input(
    label="How are you feeling today?",
    placeholder="I am ready to rock and rool!",
)

clicked = st.button("Click me")
placeholder_emotions = st.empty()
placeholder = st.empty()

def get_emotions(user_input):
    emotions = chain.run(content=user_input)
    print(f"Emotions: {emotions}")
    matches = db.similarity_search_with_score(emotions, distance_metric="cos")
    print(matches)
    doc, score = matches[0]
    iframes_html = ""
    with placeholder_emotions:
        st.write(emotions)
    with placeholder:
        embed_url = doc.metadata["embed_url"]
        iframe_html = f'<iframe src="{embed_url}" style="border:0"> </iframe>'
        st.components.v1.html(f"<div style='display:flex;flex-direction:column'>{iframe_html}</div>")


if clicked:
    get_emotions(text_input)