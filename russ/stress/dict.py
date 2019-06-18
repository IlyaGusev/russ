from enum import Enum
from typing import List, ItemsView, Set
import pickle

import pygtrie

class Stress:
    class Type(Enum):
        ANY = -1
        PRIMARY = 0
        SECONDARY = 1

    def __init__(self, position: int, stress_type: Type=Type.PRIMARY) -> None:
        self.position = position
        self.type = stress_type

    def __hash__(self):
        return hash(self.position)

    def __eq__(self, other: 'Stress'):
        return self.position == other.position and self.type == other.type

    def __str__(self):
        return str(self.position) + "\t" + str(self.type)

    def __repr__(self):
        return self.__str__()


class StressDict:
    def __init__(self):
        self.data = pygtrie.Trie()

    def save(self, dst_filename: str) -> None:
        with open(dst_filename, "wb") as f:
            pickle.dump(self.data, f, pickle.HIGHEST_PROTOCOL)

    def load(self, dump_filename: str) -> None:
        with open(dump_filename, "rb") as f:
            self.data = pickle.load(f)

    def get(self, word: str, stress_type: Stress.Type=Stress.Type.ANY) -> List[int]:
        if word in self.data:
            if stress_type == Stress.Type.ANY:
                return [stress.position for stress in self.data[word]]
            else:
                return [stress.position for stress in self.data[word] if stress.type == stress_type]
        return None

    def items(self) -> ItemsView[str, Set[Stress]]:
        return self.data.items()

    def update(self, word: str, stresses: List[Stress]) -> None:
        if word not in self.data:
            self.data[word] = set(stresses)
        else:
            self.data[word].update(stresses)

    def update_primary_only(self, word: str, stresses: List[int]) -> None:
        self.update(word, [Stress(stress, Stress.Type.PRIMARY) for stress in stresses])

    def __len__(self):
        return len(self.data)
