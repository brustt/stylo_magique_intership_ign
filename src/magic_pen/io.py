import os
import pathlib
import pickle

def make_path(file_name, *path):
    return os.path.join(*path, file_name)


def check_dir(*path):
    os.makedirs(os.path.join(*path), exist_ok=True)
    return os.path.join(*path)


def save_pickle(data, output_path):
    with open(output_path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data