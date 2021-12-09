#!/bin/bash
echo -e "This is a fuzzer for distributed DL systems"
echo -e "Work for Horovod with Pytorch."
echo -e "Mode: Horovod Elastic, multiple GPUs on multiple hosts.\n\n"

echo -e "Horovod Elastic already running, fuzzing start:, logs can be found in ./fuzzlog.log."
echo -e "Press ctl+D to stop."
# python3 ../horovod_elastic/update_hosts.py > ../horovod_elastic/fuzz.log
python3 update_hosts.py &> ./fuzz.log


echo -e "Fuzz testing ends.\n"