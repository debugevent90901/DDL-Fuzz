#!/bin/sh
echo -e "This is a script for automatically setting up a cluster for distributed deep learning system."
echo -e "Work for Horovod Elastic with Pytorch.\n\n"

echo -e "Setting up cluster environment ... "
# testing configuration:
# cloud GPU vendor: matpool.com, cluster of 4 hosts, each with one GPU
# NVIDIA Tesla P100-16GB *3 + NVIDIA GeForce RTX 3080 Ti *1 
# NIC: meth151
# horovod v0.22.1 Pytorch v1.8.1
# hosts IPs: see hosts.json

echo -e "Setting up ssh"
# use 192.168.1.11 as host
ssh root@192.168.1.10 "ssh-keygen -t rsa; \
                        ssh-copy-id root@192.168.1.11; \
                        ssh-copy-id root@192.168.1.9; \
                        ssh-copy-id root@192.168.1.8"
ssh root@192.168.1.9 "ssh-keygen -t rsa; \
                        ssh-copy-id root@192.168.1.11; \
                        ssh-copy-id root@192.168.1.10; \
                        ssh-copy-id root@192.168.1.8"
ssh root@192.168.1.8 "ssh-keygen -t rsa; \
                        ssh-copy-id root@192.168.1.11; \
                        ssh-copy-id root@192.168.1.10; \
                        ssh-copy-id root@192.168.1.9"


echo -e "Exporting environment variables"
export NCCL_SOCKET_IFNAME=meth151
export GLOO_IFACE=meth151
export NCCL_DEBUG=INFO 

ssh root@192.168.1.10 "export NCCL_SOCKET_IFNAME=meth151; \
                        export GLOO_IFACE=meth151; \
                        export NCCL_DEBUG=INFO"
ssh root@192.168.1.9 "export NCCL_SOCKET_IFNAME=meth151; \
                        export GLOO_IFACE=meth151; \
                        export NCCL_DEBUG=INFO"
ssh root@192.168.1.8 "export NCCL_SOCKET_IFNAME=meth151; \
                        export GLOO_IFACE=meth151; \
                        export NCCL_DEBUG=INFO"

echo -e "Creating discover_hosts.sh"
python3 init_hosts.py
chmod +x discover_hosts.sh
./discover_hosts
echo -e "discover_hosts.sh created.\n"

echo -e "Starting Horovod Elastic"
horovodrun -np 1 --host-discovery-script ./discover_hosts.sh --network-interface meth151 python train.py &> ./logs/elastic.log
