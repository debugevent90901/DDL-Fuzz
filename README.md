# DDLFuzz

## Introduction

**Last milestone of *CS397: Individual Study* in UIUC, FA21**

Experimental fuzz testing for distributed deep learning systems.



## Topic & Motivation

Studying Bugs in Distributed DL Systems, including Pytorch and Horovod.

After learning bugs proposed in [Pytorch PRs](https://github.com/pytorch/pytorch/pulls?q=is%3Apr+is%3Aclosed+label%3A%22oncall%3A+distributed%22) and [Horovod Issues](https://github.com/horovod/horovod/issues?q=is%3Aissue+label%3Abug+is%3Aclosed), this fuzzer aims to trigger more similar bugs in distributed training DL. 



## How to use

**./horovod_pytorch:** Fuzzer for Horovod, data parallel mode. Designed for single machine with multiple GPUs, but should also support multiple hosts with multiple GPUs.

```shell
cd horovod_pytorch
chmod +x test_horovod.sh
./test_horovod.sh
```

See logs of generated inputs and run results in ./horovod_pytorch/logs, default unittests execution time is 10.



**./horovod_elastic:** Fuzzer for Horovod Elastic, must run on distributed clusters.

Run

```shell
cd horovod_elastic
chmod +x start_elastic.sh
./start_elastic.sh
```

and 

```shell
cd horovod_elastic
chmod +x fuzz_elastic.sh
./fuzz_elastic.sh
```



concurrently in two terminal, logs will be recorded.

