
# def get_lyrics_url_from_website():
#     # https://www.disneyclips.com/lyrics/

import aiohttp
import asyncio
from bs4 import BeautifulSoup

from typing import List, TypedDict, Tuple, Optional

class Lyric(TypedDict):
    name: str 
    text: str

class Movie(TypedDict):
    title: str 
    lyrics: List[Lyric]


URL = "https://www.disneyclips.com/lyrics/"


async def get_lyrics_urls_from_movie_url(url: str, session: aiohttp.ClientSession) -> Optional[Tuple[str, str]]:
    async with session.get(url) as response:
        html = await response.text()
        soup = BeautifulSoup(html, 'html.parser')
        table = soup.find('table', {'class': 'songs'})
        names_and_urls = None
        if table:
            links = table.find_all('a')
            names_and_urls = []
            for link in links:
                names_and_urls.append((link.text,  f"{URL}/{link.get('href')}"))
        return names_and_urls

async def get_lyric_from_lyric_url(url: str, name: str, session: aiohttp.ClientSession) -> Lyric:
    async with session.get(url) as response:
        html = await response.text()
        soup = BeautifulSoup(html, 'html.parser')
        div = soup.find('div', {'id': 'cnt'}).find('div', {'class': 'main'})
        paragraphs = div.find_all('p')
        text = ""
        for p in paragraphs:
            text += p.text
        return text



async def get_movie_names_and_urls(session: aiohttp.ClientSession) -> List[Tuple[str, str]]:
    async with session.get(URL) as response:
        html = await response.text()
        soup = BeautifulSoup(html, 'html.parser')
        links = soup.find('div', {'id': 'cnt'}).find('div', {'class': 'main'}).find_all('a')
        movie_names_and_urls = [(link.text, f"{URL}/{link.get('href')}") for link in links]
        return movie_names_and_urls
       



async def main():
    async with aiohttp.ClientSession() as session:
        names_and_urls = await get_movie_names_and_urls(session)
        data = await asyncio.gather(*[asyncio.create_task(get_lyrics_urls_from_movie_url(names, url, session)) for (names, url) in names_and_urls])

loop = asyncio.get_event_loop()
loop.run_until_complete(main())