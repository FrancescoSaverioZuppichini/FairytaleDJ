from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

load_dotenv()
import os
import json
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings

from data import load_db
from names import DATASET_ID, MODEL_ID
import random


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
    input_variables=["songs", "user_input"],
    template=Path("prompts/bot_with_summary.prompt").read_text(),
)

    llm = ChatOpenAI(temperature=0.7)

    chain = LLMChain(llm=llm, prompt=prompt)

    with open("data/emotions_with_spotify_url.json", "r") as f:
        data = json.load(f)
    
    movies_and_names_to_songs = {}

    songs_str = ""

    for movie, songs in data.items():
        for song in songs:
            movie_and_name = f"{movie};{song['name']}".lower()
            songs_str += f"{movie_and_name}:{song['text']}\n"
            movies_and_names_to_songs[movie_and_name] = song

    return db, chain, movies_and_names_to_songs, songs_str

db, chain, movies_and_names_to_songs, songs_str = init()

st.title("Disney song for you")

text_input = st.text_input(
    label="How are you feeling today?",
    placeholder="I am ready to rock and rool!",
)

clicked = st.button("Click me")
placeholder_emotions = st.empty()
placeholder = st.empty()

def get_emotions(songs_str, user_input):
    res = chain.run(songs=songs_str, user_input=user_input)
    song_key = random.choice(eval(res))
    doc = movies_and_names_to_songs[song_key.lower()]
    print(f"Reply: {res}, chosen: {song_key}")
    with placeholder:
        embed_url = doc["embed_url"]
        iframe_html = f'<iframe src="{embed_url}" style="border:0"> </iframe>'
        st.components.v1.html(f"<div style='display:flex;flex-direction:column'>{iframe_html}</div>")

if clicked:
    get_emotions(songs_str, text_input)