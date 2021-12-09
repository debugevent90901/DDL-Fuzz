#!/bin/bash
echo -e "This is a fuzzer for distributed DL systems"
echo -e "Work for Horovod with Pytorch."
echo -e "Mode: Horovod data parallel, multiple GPUs on single machine.\n\n"


echo -e "Fuzzing Start: generate inputs in ../horovod/inputs."
for seed_file in $(ls ../horovod/seeds)
do  
    echo -e ${seed_file}
    python3 ../horovod/mutate.py ../horovod/seeds/${seed_file} > ../horovod/logs/mutate.log
done
echo -e "Inputs generation finished, see mutate.log in ../horovod/logs for details. \n"


echo -e "Start to run each input, logs can be found in ../horovod/logs."
for input_file in $(ls ../horovod/inputs)
do
    echo -e "Test ../horovod/inputs/${input_file} start. "
    # horovodrun -np 4 python3 ../horovod/inputs/${input_file} > ../horovod/logs/${input_file}.output
    echo -e "Test ../horovod/inputs/${input_file} finished, see corresponding log in ../horovod/logs for detail."
done


echo -e "Fuzzing ends.\n"