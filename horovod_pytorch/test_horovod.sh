#!/bin/bash
echo -e "This is a fuzzer for distributed DL systems"
echo -e "Work for Horovod with Pytorch."
echo -e "Mode: Horovod data parallel, multiple GPUs on single machine.\n\n"

# testing configuration:
# cloud GPU vendor: matpool.com, single host, NVIDIA Tesla K80 x 4
# horovod v0.22.1 Pytorch v1.8.1

echo -e "Start fuzzing ... "

int=1

# number of executing unittests: 10
while (( $int <= 10 ))
do
    echo -e executing unittest $int
    python3 fuzz.py $int
    # horovodrun -np 4 pytest -v template.py
    # horovodrun -np 4 pytest -v template.py &> ./logs/run$int.log 
    python3 test.py &> ./logs/run$int.log 
    let "int++"
done

echo -e "Fuzzing ends.\n"