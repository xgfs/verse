from typing import Dict

import numpy as np


class Embedding(object):
    def __init__(self, embedding_path: str, dimensions: int, index_path: str = None):
        self.dimensions = dimensions
        self.embeddings = self.load_embeddings(embedding_path)
        self.index: Dict[str, int] = {}
        if index_path:
            self.load_index(index_path)

    def load_embeddings(self, file_name: str) -> np.ndarray:
        print("Loading embeddings...")
        embeddings = np.fromfile(file_name, dtype=np.float32)
        length = embeddings.shape[0]
        assert length % self.dimensions == 0, f"The number of floats ({length}) in the embeddings is not divisible by" \
                                              f"the number of dimensions ({self.dimensions})!"
        embedding_shape = [int(length / self.dimensions), self.dimensions]
        embeddings = embeddings.reshape(embedding_shape)
        print(f"Done loading embeddings (shape: {embeddings.shape}).")
        return embeddings

    def load_index(self, index_path: str) -> None:
        print("Loading uri index...")
        with open(index_path, "r") as file:
            for line in [line.strip() for line in file.readlines()]:
                index, uri = line.split(",", 1)
                self.index[uri] = int(index)
        print(f"Done loading {len(self.index)} items.")

    def __getitem__(self, item) -> np.ndarray:
        if self.index and isinstance(item, str):
            return self.embeddings[self.index[item]]
        return self.embeddings[item]
