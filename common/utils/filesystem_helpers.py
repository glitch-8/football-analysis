import os

CURR_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.abspath(os.path.join(CURR_PATH, '../..'))


def get_path(path: str) -> str:
    return os.path.abspath(os.path.join(ROOT_PATH, path))