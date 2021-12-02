#!/bin/bash
echo -e "delete previous records of the fuzzer.\n"

rm -rf ../horovod/logs
rm -rf ../horovod/inputs

mkdir ../horovod/logs
mkdir ../horovod/inputs