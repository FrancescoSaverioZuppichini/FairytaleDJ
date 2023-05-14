from dotenv import load_dotenv

load_dotenv()
import json
import os

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import DeepLake

from names import DATASET_ID, MODEL_ID


def create_db(dataset_path: str, json_filepath: str) -> DeepLake:
    with open(json_filepath, "r") as f:
        data = json.load(f)

    texts = []
    metadatas = []

    for movie, lyrics in data.items():
        for lyric in lyrics:
            texts.append(lyric["text"])
            metadatas.append(
                {
                    "movie": movie,
                    "name": lyric["name"],
                    "embed_url": lyric["embed_url"],
                }
            )

    embeddings = OpenAIEmbeddings(model=MODEL_ID)

    db = DeepLake.from_texts(
        texts, embeddings, metadatas=metadatas, dataset_path=dataset_path
    )

    return db


def load_db(dataset_path: str, *args, **kwargs) -> DeepLake:
    db = DeepLake(dataset_path, *args, **kwargs)
    return db


if __name__ == "__main__":
    dataset_path = f"hub://{os.environ['ACTIVELOOP_ORG_ID']}/{DATASET_ID}"
    create_db(dataset_path, "data/emotions_with_spotify_url.json")
