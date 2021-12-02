#!/bin/bash
echo -e "This is a fuzzer for distributed DL systems"
echo -e "Work for Horovod with Pytorch."
echo -e "Mode: Horovod Elastic, multiple GPUs on multiple hosts.\n\n"


echo -e "Creating discover_hosts.sh"
python3 ../horovod_elastic/init_hosts.py
echo -e "discover_hosts.sh created.\n"

echo -e "Init Horovod Elastic"
chmod +x ../horovod_elastic/discover_hosts.sh
# horovodrun -np 8 --host-discovery-script discover_hosts.sh python train.py
# horovodrun -np 8 --min-np 4 --max-np 12 --host-discovery-script discover_hosts.sh python train.py
echo -e "Horovod Elastic started.\n"

echo -e "Fuzzing start:, logs can be found in ../horovod_elastic/fuzzlog.log."
echo -e "Press ctl+D to stop."
python3 ../horovod_elastic/updateHosts.py > ../horovod_elastic/fuzz.log


echo -e "Fuzz testing ends.\n"