

import random, time

TIME_UPPER = 8
TIME_LOWER = 5

# TODO: finish this
def kill_worker():
    while(1):
        random_time = random.randint(TIME_LOWER, TIME_UPPER)
        time.sleep(random_time)
        random_host = random.randint(1, 4)
        print("Worker %d has been killed." %random_host)