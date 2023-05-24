import queue
import threading

l = threading.Lock()

def worker(q):
    while True:
        item = q.get()
        if item is None:
            break
        with l:
            print("Processing", item)

q = queue.Queue()

threads = []
for i in range(5):
    t = threading.Thread(target=worker, args=(q,))
    t.start()
    threads.append(t)

for item in range(20):
    q.put(item)

for i in range(5):
    q.put(None)

for t in threads:
    t.join()
