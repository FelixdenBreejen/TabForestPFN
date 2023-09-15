from pathlib import Path
import torch.multiprocessing as mp


class Writer():

    def __init__(self, queue: mp.Queue) -> None:
        self.queue = queue

    def write(self, path: Path, txt: str, mode: str) -> None:
        self.queue.put((path, txt, mode))


def file_writer(queue: mp.Queue) -> None:

    while True:
        path, txt, mode = queue.get()
        with open(path, mode) as f:
            f.write(txt)