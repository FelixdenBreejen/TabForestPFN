from pathlib import Path
import torch.multiprocessing as mp


class Writer():

    def __init__(self, queue: mp.Queue) -> None:
        self.queue = queue

    def write(self, path: Path, txt: str) -> None:
        self.queue.put((path, txt))


def file_writer(queue: mp.Queue) -> None:

    while True:
        path, txt = queue.get()
        with open(path, 'a') as f:
            f.write(txt)