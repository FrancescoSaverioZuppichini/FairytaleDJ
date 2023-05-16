import os
from typing import List, TypedDict
from uuid import uuid4

import redis


class UserInput(TypedDict):
    text: str
    emotions: str
    songs: List[str]


class RedisStorage:
    def __init__(self, host: str, password: str):
        self._client = redis.Redis(host=host, port="34307", password=password, ssl=True)

    def store(self, data: UserInput) -> bool:
        uid = uuid4()
        response = self._client.json().set(f"data:{uid}", "$", data)
        return response
