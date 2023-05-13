from dotenv import load_dotenv

load_dotenv()
import json
import os
from pathlib import Path

import deeplake
import numpy as np
import openai

# https://www.disneyclips.com/lyrics/
DATASET_NAME = "disney-lyrics"
model_id = "text-embedding-ada-002"
dataset_path = f"hub://{os.environ['ACTIVELOOP_ORG_ID']}/{DATASET_NAME}"
print(dataset_path)
runtime = {"db_engine": True}

with open("lyrics.json", "rb") as f:
    lyrics = json.load(f)["lyrics"]

# embeddings = [el["embedding"] for el in openai.Embedding.create(input=lyrics, model=model_id)['data']]

# embeddings_np = np.array(embeddings)
# np.save("embeddings.npy", embeddings_np)

embeddings_np = np.load("embeddings.npy")

print(embeddings_np.shape)


# ds = deeplake.empty(dataset_path, runtime=runtime, overwrite=True)

# # https://docs.deeplake.ai/en/latest/Htypes.html
# with ds:
#     ds.create_tensor("embedding", htype="embedding", dtype=np.float32, exist_ok=True)
#     ds.extend({ "embedding": embeddings_np.astype(np.float32)})
#     ds.summary()

search_term = "Let's get down to business"

embedding = openai.Embedding.create(input=search_term, model="text-embedding-ada-002")[
    "data"
][0]["embedding"]

# Format the embedding as a string, so it can be passed in the REST API request.
embedding_search = ",".join([str(item) for item in embedding])

# embedding_search = ",".join([str(item) for item in embeddings_np[0].tolist()])
# print(embedding_search)


ds = deeplake.load(dataset_path)

# print(embedding_search)
query = f'select * from (select l2_norm(embedding - ARRAY[{embedding_search}]) as score from "{dataset_path}") order by score desc limit 5'
with open("foo.txt", "w") as f:
    f.write(query)
query_res = ds.query(query)
print(query_res)
