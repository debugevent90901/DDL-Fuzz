#!/bin/bash

while true
do
        ../horovod_elastic/discover_hosts.sh
        sleep 3
        echo -e "\n\n"
done