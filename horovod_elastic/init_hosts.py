
import json
import numpy as np


def load_all_hosts(hostfile):
    with open(hostfile, "r") as f:
        hosts = json.load(f)
        f.close()
    print("available hosts: ", hosts)
    
    all_hosts = {}
    for id, host in enumerate(hosts, start=0):
        all_hosts[id] = (host, hosts[host])
    
    return all_hosts


def write_hosts_to_script(ids, hosts, mode="override", action=None):
    if mode == "override":
        lines = ["#!/bin/bash\n"]
        for i in ids:
            line = 'echo -e '
            host, slots = hosts[i][0], str(hosts[i][1])
            line += '"' + host + ':' + slots + '"\n'
            lines.append(line)
        
    if mode == "update":
        with open('./discover_hosts.sh', "r") as f:
            lines = f.readlines()
            f.close()
        if action == 1:
            # append in the end
            for i in ids:
                line = 'echo -e '
                host, slots = hosts[i][0], str(hosts[i][1])
                line += '"' + host + ':' + slots + '"\n'
                lines.append(line)
        if action == -1:
            # remove
            tmp = ["#!/bin/bash\n"]
            for i in lines[1:]:
                if int(i[-5])-1 not in ids:
                    tmp.append(i)
            lines = tmp
    
    with open("./discover_hosts.sh", "w") as f:
        f.writelines(lines)
        f.close()


if __name__ == "__main__":
    all_hosts = load_all_hosts('./hosts.json')
    num_all_hosts = len(all_hosts)
    num_init_hosts = np.random.randint(1, num_all_hosts+1)
    # print(num_init_hosts)
    init_hosts = np.random.choice(range(num_all_hosts), num_init_hosts, replace=False)
    # print(init_hosts)

    write_hosts_to_script(init_hosts, all_hosts, mode="override")