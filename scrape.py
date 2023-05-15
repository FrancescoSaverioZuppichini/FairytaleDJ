import asyncio
import json
from collections import defaultdict
from itertools import chain
from typing import List, Optional, Tuple, TypedDict

import aiohttp
from bs4 import BeautifulSoup

"""
This file scrapes disney songs + lyrics from "https://www.disneyclips.com/lyrics/"
"""

URL = "https://www.disneyclips.com/lyrics/"


async def get_lyrics_names_and_urls_from_movie_url(
    movie_name: str, url: str, session: aiohttp.ClientSession
) -> List[Tuple[str, str]]:
    async with session.get(url) as response:
        html = await response.text()
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table", {"class": "songs"})
        names_and_urls = []
        if table:
            links = table.find_all("a")
            names_and_urls = []
            for link in links:
                names_and_urls.append(
                    (movie_name, link.text, f"{URL}/{link.get('href')}")
                )
        return names_and_urls


async def get_lyric_from_lyric_url(
    movie_name: str, lyric_name: str, url: str, session: aiohttp.ClientSession
) -> str:
    async with session.get(url) as response:
        html = await response.text()
        soup = BeautifulSoup(html, "html.parser")
        div = soup.find("div", {"id": "cnt"}).find("div", {"class": "main"})
        paragraphs = div.find_all("p")
        text = ""
        # first <p> has the lyric
        p = paragraphs[0]
        for br in p.find_all("br"):
            br.replace_with(". ")
        for span in p.find_all("span"):
            span.decompose()
        text += p.text

        return (movie_name, lyric_name, text)


async def get_movie_names_and_urls(
    session: aiohttp.ClientSession,
) -> List[Tuple[str, str]]:
    async with session.get(URL) as response:
        html = await response.text()
        soup = BeautifulSoup(html, "html.parser")
        links = (
            soup.find("div", {"id": "cnt"}).find("div", {"class": "main"}).find_all("a")
        )
        movie_names_and_urls = [
            (link.text, f"{URL}/{link.get('href')}") for link in links
        ]
        return movie_names_and_urls


async def scrape_disney_lyrics():
    async with aiohttp.ClientSession() as session:
        data = await get_movie_names_and_urls(session)
        data = await asyncio.gather(
            *[
                asyncio.create_task(
                    get_lyrics_names_and_urls_from_movie_url(*el, session)
                )
                for el in data
            ]
        )
        data = await asyncio.gather(
            *[
                asyncio.create_task(get_lyric_from_lyric_url(*data, session))
                for data in chain(*data)
            ]
        )

        result = defaultdict(list)

        for movie_name, lyric_name, lyric_text in data:
            result[movie_name].append({"name": lyric_name, "text": lyric_text})

        with open("data/lyrics.json", "w") as f:
            json.dump(result, f)


loop = asyncio.get_event_loop()
loop.run_until_complete(scrape_disney_lyrics())
