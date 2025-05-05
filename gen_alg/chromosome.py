from __future__ import annotations
import random


class BinaryChromosome:
    def __init__(self, gene: str):
        if not set(gene) <= {"0", "1"}:
            raise ValueError("Gene must consist only of 0/1.")
        self.gene = gene

    @staticmethod
    def random(bits: int) -> "BinaryChromosome":
        return BinaryChromosome("".join(random.choice("01") for _ in range(bits)))

    @staticmethod
    def decode_bits(bits: list[int] | str, low: float, high: float) -> float:
        if isinstance(bits, str):
            bin_str = bits
        else:
            bin_str = "".join(str(b) for b in bits)
        m = len(bin_str)
        dec = int(bin_str, 2)
        return low + dec * (high - low) / (2**m - 1)

    def flip(self, idx: int) -> None:
        g = list(self.gene)
        g[idx] = "1" if g[idx] == "0" else "0"
        self.gene = "".join(g)

    def copy(self) -> "BinaryChromosome":
        return BinaryChromosome(self.gene)

    def __len__(self):
        return len(self.gene)

    def __repr__(self):
        return f"BinaryChromosome('{self.gene}')"


class RealChromosome(list):
    def copy(self) -> "RealChromosome":
        return RealChromosome(self)
