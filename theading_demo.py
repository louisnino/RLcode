
import tensorflow as tf
import threading
import queue

N_WORKER = 4            #worker的数量
QUEUE = queue.Queue()   #队列，用于储存数据
EP_MAX = 10             #执行EP
EP_LEN = 200            #每个EP的最大步数
MIN_BATCH_SIZE = 10     #每个batch的大小


class Worker():
    #工人对象的id。该程序只是模拟，所以在填入数据的时候，会直接把wid放入队列表示该工人产生的数据。
    def __init__(self,wid):
        self.wid = wid              #工人id

    def work(self):
        global GLOBAL_EP, GLOBAL_UPDATE_COUNTER
        
        #判断是否所有线程都应该停止了。
        while not COORD.should_stop():
            
            for _ in range(EP_LEN):                 #开始新的EP

                #if not ROLLING_EVENT.is_set():      #如果有其他worker线程已经被阻塞，那么其他线程也需要在这等待。  
                ROLLING_EVENT.wait()
                    
                QUEUE.put(self.wid)
                '''
                这里做了简化，直接把worker的id当做和环境互动产生的数据放入队列中。 
                实际上，这里会用buffer记录智能体和环境互动产生的数据。当数据大于MIN_BATCH_SIZE才开始整理数据。
                '''              
                GLOBAL_UPDATE_COUNTER += 1          #GLOBAL_UPDATE_COUNTER+1:表示有智能体走了一步了


                if GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE: #更新大于
                    '''
                    这里可以插入整理数据部分
                    '''
                    ROLLING_EVENT.clear()
                    UPDATE_EVENT.set()
                        
                if GLOBAL_EP >= EP_MAX:          #更新10次
                    COORD.request_stop()
                    break

            


class PPO(object):
    
    def update(self):
        global GLOBAL_UPDATE_COUNTER
        
        #判断是否所有线程都应该停止了。
        while not COORD.should_stop():
            if GLOBAL_EP <= EP_MAX:
                UPDATE_EVENT.wait()

                '''
                这里用输出表示更新
                '''           
                print("====update====")
                print("GLOBAL_EP",GLOBAL_EP)
                print("GLOBAL_UPDATE_COUNTER:",GLOBAL_UPDATE_COUNTER)
                print("update_old_pi")
                print("Queuesize:",QUEUE.qsize())
                print([QUEUE.get() for _ in range(QUEUE.qsize())])
                print("update Critic")
                print("update Actor")
                print("=====END======")

                GLOBAL_UPDATE_COUNTER = 0

                UPDATE_EVENT.clear()
                ROLLING_EVENT.set()       

if __name__ == "__main__":
    #创建worker对象
    #做法1：
    workers = []
    for i in range(N_WORKER):
        worker = Worker(i)
        workers.append(worker)
    #做法2：
    #workers = [Worker(wid=i) for i in range(N_WORKER)]

    #创建PPO对象
    GLOBAL_PPO = PPO()

    #新建两个event:UPDATE_EVENT,ROLLING_EVENT
    #把UPDATE_EVENT的信号设置为阻塞
    #把ROLLING_EVENT的信号设置为就绪
    UPDATE_EVENT,ROLLING_EVENT = threading.Event(), threading.Event()
    UPDATE_EVENT.clear()
    ROLLING_EVENT.set()

    #定义两个全局变量
    #GLOBAL_UPDATE_COUNTER：每次更新+1
    #GLOBAL_STEP：
    GLOBAL_UPDATE_COUNTER, GLOBAL_EP = 0, 0
    threads = []

    #创建协调器
    COORD = tf.train.Coordinator()

    #开启rolling线程
    for worker in workers:                          #三个rolling线程
        t = threading.Thread(target=worker.work)    #线程的功能就是执行work函数
        t.start()                                   
        threads.append(t)
    
    #开启update线程
    threads.append(threading.Thread(target=GLOBAL_PPO.update,)) #update线程执行PPO的update函数
    threads[-1].start()                                         #启动最后加入的线程，就是update线程
    
    #加入协调器
    COORD.join(threads)