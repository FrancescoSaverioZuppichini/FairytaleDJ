"""
This script will keep only the songs that are in the Spotify "Disney Hits" playlist
"""
from dotenv import load_dotenv

load_dotenv()
import json
from collections import defaultdict

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

name = "Disney hits"

spotify = spotipy.Spotify(auth_manager=SpotifyClientCredentials())
results = spotify.search(q="playlist:" + name, type="playlist", limit=5)
items = results["playlists"]["items"]

uri = "spotify:playlist:37i9dQZF1DX8C9xQcOrE6T"
playlist = spotify.playlist(uri)

with open("data/lyrics.json", "r") as f:
    data = json.load(f)

spotify_tracks = {}

for item in playlist["tracks"]["items"]:
    track = item["track"]
    track_name = track["name"].lower().split("-")[0].strip()
    print(track_name)
    spotify_tracks[track_name] = {
        "id": track["id"],
        "embed_url": f"https://open.spotify.com/embed/track/{track['id']}?utm_source=generator",
    }

# here we add only songs that are in the Disney spotify playlist

data_filtered = defaultdict(list)
tot = 0
for movie, lyrics in data.items():
    for lyric in lyrics:
        name = lyric["name"].lower()
        if name in spotify_tracks:
            data_filtered[movie].append(
                {**lyric, **{"embed_url": spotify_tracks[name]["embed_url"]}}
            )
            tot += 1
print(tot)

with open("data/lyrics_with_spotify_url.json", "w") as f:
    json.dump(data_filtered, f)
