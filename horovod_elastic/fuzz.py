

from update_hosts import update_hosts
from kill_worker import kill_worker
import threading, time


if __name__ == "__main__":
    try:
        thread_update_hosts = threading.Thread(target=update_hosts)
        thread_kill_workers = threading.Thread(target=kill_worker)
        
        thread_update_hosts.start()
        thread_kill_workers.start()
        thread_update_hosts.join()
        thread_kill_workers.join()
    except:
        print(11)

