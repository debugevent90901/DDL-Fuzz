#!/bin/bash
echo -e "This is a script for automatically setting up a cluster for distributed deep learning system."
echo -e "Work for Horovod Elastic with Pytorch."

echo -e "Creating discover_hosts.sh"
python3 init_hosts.py
chmod +x discover_hosts.sh
./discover_hosts

echo -e "discover_hosts.sh created.\n"

echo -e "Starting Horovod Elastic"
# testing configuration:
# cloud GPU vendor: matpool.com, cluster of 4 hosts, each with one GPU
# NVIDIA Tesla P100-16GB *3 + NVIDIA GeForce RTX 3080 Ti *1 
# NIC: meth151
# horovod v0.22.1
# hosts IPs: see hosts.json
horovodrun -np 1 --host-discovery-script ./discover_hosts.sh --network-interface meth151 python train.py
