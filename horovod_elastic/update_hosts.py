from init_hosts import load_all_hosts, write_hosts_to_script
import time, subprocess
import numpy as np

UPDATE_TIME_INTERVAL = 8
PROBABLITY_TO_KILL = 0.75
LOCAL_HOST_ID = 1
KILL_WORKER_ENABLE = 0

def load_curr_hosts():
    child = subprocess.Popen(['./discover_hosts.sh'], stdout=subprocess.PIPE)
    stdout, _ = child.communicate()
    _curr_hosts = str(stdout, encoding='utf-8').split("\n")[:-1] 
    curr_hosts = {}
    # print(curr_hosts)
    for i in _curr_hosts:
        curr_hosts[int(i[-3])-1] = (i[:-2], int(i[-1]))

    print("current working hosts: ", curr_hosts)
    return curr_hosts


def mutate(curr_hosts, all_hosts, action):
    if action == -1:
        # remove workers
        num_to_remove = np.random.randint(1, len(curr_hosts))
        ids = np.random.choice(list(curr_hosts.keys()), num_to_remove, replace=False)
        print("remove %d hosts: " %num_to_remove, ids)
    if action == 1:
        num_to_add = np.random.randint(1, len(all_hosts)-len(curr_hosts)+1)
        ids = np.random.choice(list(set(all_hosts.keys()).difference(set(curr_hosts.keys()))), num_to_add, replace=False)
        print("add %d hosts: " %num_to_add, ids)

    return ids

def kill_worker(host):
    subprocess.run(['ssh', 'root@'+host, '"pkill -f python'])


# def update_hosts():
if __name__ == "__main__":
    all_hosts = load_all_hosts('./hosts.json')
    # print(all_hosts)
    curr_hosts = load_curr_hosts()
    # print(curr_hosts)
    # i = 10
    while(i>0):
    # while(1):
        time.sleep(UPDATE_TIME_INTERVAL)
        # i -= 1
        if KILL_WORKER_ENABLE:
            # only kill a worker if some probablity requirement is met
            p = np.random.uniform()
            if p >= PROBABLITY_TO_KILL:
                candidants = list(curr_hosts.keys())
                if LOCAL_HOST_ID in candidants:
                    candidants.remove(LOCAL_HOST_ID)
                worker_to_kill = np.random.choice(candidants)
                host = curr_hosts[worker_to_kill][0]
                kill_worker(host)

        curr_hosts = load_curr_hosts()
        # print(curr_hosts)
        if (len(curr_hosts) == len(all_hosts)):
            # in this case, mutation must remove workers
            ids = mutate(curr_hosts, all_hosts, action=-1)
            write_hosts_to_script(ids, all_hosts, mode="update", action=-1)
            # print(load_curr_hosts())
            continue
        elif len(curr_hosts) == 1:
            # in this case, mutation must add workers
            ids = mutate(curr_hosts, all_hosts, action=1)
            write_hosts_to_script(ids, all_hosts, mode="update", action=1)
            # print(load_curr_hosts())
            continue
        else:
            # now mutation can either add or remove workers
            # action=1 implies addition, action=-1 implies removal
            action = np.random.choice([1, -1], 1)
            ids = mutate(curr_hosts, all_hosts, action)
            write_hosts_to_script(ids, all_hosts, mode="update", action=action)
            # print(load_curr_hosts())
            continue
            
