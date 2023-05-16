from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

load_dotenv()
import os
from typing import List, Tuple

import numpy as np
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document

from data import load_db
from names import DATASET_ID, MODEL_ID
from storage import RedisStorage, UserInput
from utils import weighted_random_sample


class RetrievalType:
    FIRST_MATCH = "first-match"
    POOL_MATCHES = "pool-matches"


Matches = List[Tuple[Document, float]]
USE_STORAGE = os.environ.get("USE_STORAGE", "True").lower() in ("true", "t", "1")

print("USE_STORAGE", USE_STORAGE)


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

    storage = RedisStorage(
        host=os.environ["UPSTASH_URL"], password=os.environ["UPSTASH_PASSWORD"]
    )
    prompt = PromptTemplate(
        input_variables=["user_input"],
        template=Path("prompts/bot.prompt").read_text(),
    )

    llm = ChatOpenAI(temperature=0.3)

    chain = LLMChain(llm=llm, prompt=prompt)

    return db, storage, chain


# Don't show the setting sidebar
if "sidebar_state" not in st.session_state:
    st.session_state.sidebar_state = "collapsed"

st.set_page_config(initial_sidebar_state=st.session_state.sidebar_state)


db, storage, chain = init()

st.title("FairytaleDJ 🎵🏰🔮")
st.markdown(
    """
*<small>Made with [DeepLake](https://www.deeplake.ai/) 🚀 and [LangChain](https://python.langchain.com/en/latest/index.html) 🦜⛓️</small>*

💫 Unleash the magic within you with our enchanting app, turning your sentiments into a Disney soundtrack! 🌈 Just express your emotions, and embark on a whimsical journey as we tailor a Disney melody to match your mood. 👑💖""",
    unsafe_allow_html=True,
)
how_it_works = st.expander(label="How it works")

text_input = st.text_input(
    label="How are you feeling today?",
    placeholder="I am ready to rock and rool!",
)

run_btn = st.button("Make me sing! 🎶")
with how_it_works:
    st.markdown(
        """
The application follows a sequence of steps to deliver Disney songs matching the user's emotions:
- **User Input**: The application starts by collecting user's emotional state through a text input.
- **Emotion Encoding**: The user-provided emotions are then fed to a Language Model (LLM). The LLM interprets and encodes these emotions.
- **Similarity Search**: These encoded emotions are utilized to perform a similarity search within our [vector database](https://www.deeplake.ai/). This database houses Disney songs, each represented as emotional embeddings.
- **Song Selection**: From the pool of top matching songs, the application randomly selects one. The selection is weighted, giving preference to songs with higher similarity scores.
- **Song Retrieval**: The selected song's embedded player is displayed on the webpage for the user. Additionally, the LLM interpreted emotional state associated with the chosen song is displayed.
"""
    )


placeholder_emotions = st.empty()
placeholder = st.empty()


with st.sidebar:
    st.text("App settings")
    filter_threshold = st.slider(
        "Threshold used to filter out low scoring songs",
        min_value=0.0,
        max_value=1.0,
        value=0.8,
    )
    max_number_of_songs = st.slider(
        "Max number of songs we will retrieve from the db",
        min_value=5,
        max_value=50,
        value=20,
        step=1,
    )
    number_of_displayed_songs = st.slider(
        "Number of displayed songs", min_value=1, max_value=4, value=2, step=1
    )


def filter_scores(matches: Matches, th: float = 0.8) -> Matches:
    return [(doc, score) for (doc, score) in matches if score > th]


def normalize_scores_by_sum(matches: Matches) -> Matches:
    scores = [score for _, score in matches]
    tot = sum(scores)
    return [(doc, (score / tot)) for doc, score in matches]


def get_song(user_input: str, k: int = 20):
    emotions = chain.run(user_input=user_input)
    matches = db.similarity_search_with_score(emotions, distance_metric="cos", k=k)
    # [print(doc.metadata['name'], score) for doc, score in matches]
    docs, scores = zip(
        *normalize_scores_by_sum(filter_scores(matches, filter_threshold))
    )
    choosen_docs = weighted_random_sample(
        np.array(docs), np.array(scores), n=number_of_displayed_songs
    ).tolist()
    return choosen_docs, emotions


def set_song(user_input):
    if user_input == "":
        return
    # take first 120 chars
    user_input = user_input[:120]
    docs, emotions = get_song(user_input, k=max_number_of_songs)
    print(docs)
    songs = []
    with placeholder_emotions:
        st.markdown("Your emotions: `" + emotions + "`")
    with placeholder:
        iframes_html = ""
        for doc in docs:
            name = doc.metadata["name"]
            print(f"song = {name}")
            songs.append(name)
            embed_url = doc.metadata["embed_url"]
            iframes_html += (
                f'<iframe src="{embed_url}" style="border:0;height:100px"> </iframe>'
            )

        st.markdown(
            f"<div style='display:flex;flex-direction:column'>{iframes_html}</div>",
            unsafe_allow_html=True,
        )

        if USE_STORAGE:
            success_storage = storage.store(
                UserInput(text=user_input, emotions=emotions, songs=songs)
            )
            if not success_storage:
                print("[ERROR] was not able to store user_input")


if run_btn:
    set_song(text_input)
