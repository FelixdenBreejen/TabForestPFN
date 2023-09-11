

def writer(queue) -> None:

    while True:
        path, txt = queue.get()
        with open(path, 'a') as f:
            f.write(txt)