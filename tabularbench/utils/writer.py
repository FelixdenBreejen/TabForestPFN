from pathlib import Path
import torch.multiprocessing as mp


class Writer():

    def __init__(self, queue: mp.JoinableQueue) -> None:
        self.queue = queue

    def write(self, path: Path, txt: str, mode: str) -> None:
        self.queue.put((path, txt, mode))
        self.queue.join()

class StandardWriter():

    def __init__(self) -> None:
        pass

    def write(self, path: Path, txt: str, mode: str) -> None:
        with open(path, mode) as f:
            f.write(txt)


def file_writer(queue: mp.JoinableQueue) -> None:

    while True:
        path, txt, mode = queue.get()
        with open(path, mode) as f:
            f.write(txt)
        queue.task_done()