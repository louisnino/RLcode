import queue

QUEUE = queue.Queue()

def q_demo():
    for i in range(10):
        QUEUE.put(i)
    for i in range(10):
        print(QUEUE.get())

if __name__=="__main__":
    q_demo()