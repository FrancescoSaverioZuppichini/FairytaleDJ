from dotenv import load_dotenv
load_dotenv() 

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from pathlib import Path
from langchain.chat_models import ChatOpenAI
import json
from collections import defaultdict
from pprint import pprint

prompt = PromptTemplate(
    input_variables=["song"],
    template=Path("prompts/summary.prompt").read_text(),
)

llm = ChatOpenAI(temperature=0)

chain = LLMChain(llm=llm, prompt=prompt)

with open("/home/zuppif/Documents/Work/ActiveLoop/ai-shazam/data/lyrics_with_spotify_url.json", "r") as f:
    data = json.load(f)

lyrics_summaries = {}

for movie, lyrics in data.items():
    for lyric in lyrics:
        print(f"Creating summary for {lyric['name']}")
        summary = chain.run(song=lyric['text'])
        lyrics_summaries[lyric['name'].lower()] = {"summary": summary,  "embed_url": lyric["embed_url"] }

with open("/home/zuppif/Documents/Work/ActiveLoop/ai-shazam/data/lyrics_with_spotify_url_and_summary.json", "w") as f:
    json.dump(lyrics_summaries, f)

pprint(lyrics_summaries)