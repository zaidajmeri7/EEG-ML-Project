import pickle
from pathlib import Path

file_path = Path("data/deap/s01.dat")

with open(file_path, "rb") as f:
    subject = pickle.load(f, encoding="latin1")

print("Keys:", subject.keys())
print("Data shape:", subject["data"].shape)
print("Labels shape:", subject["labels"].shape)