import os, sys, shutil
from config import *


seed_file = sys.argv[1]
src = "../seeds/" + seed_file

for id in range(NUM_MUTATION):
    # first make a copy of seed 
    dest = "../inputs/input_" + seed_file[:-3] + "_generated_id" + str(id)

    with open(src, "r") as f:
        lines = f.readlines()
        f.close()

    # then start to mutate 
    # mutate for horovod elastic
    if ("elastic" in seed_file):
        pass

    # mutate for horovod with pytorch
    else:
        # id1~id2: train function in seed, can be modified
        # id2~id3: main entry of the file, can be modified
        id1 = lines.index("# TRAIN") + 1
        id2 = lines.index("# ENTRY") + 1
        id3 = len(lines)



    # finally write to inputs folder
    with open(dest, "w") as f:
        f.writelines(lines)
        f.close()
    print("Generation for " + seed_file[:-3] + " done.\n")