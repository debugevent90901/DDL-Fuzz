import sys
import numpy as np
from torch.nn.functional import log_softmax


def generate_inputs(id):
    torch_seed = np.random.randint(1000, 10000)
    numpy_seed = np.random.randint(1000, 10000)
    random_int1 = np.random.randint(10, 20)
    random_int2 = np.random.randint(3, 8)
    N = np.random.randint(10, 100)
    D_in = np.random.randint(10, 100)
    H = np.random.randint(10, 100)
    D_out = np.random.randint(10, 100)
    arg4 = str(N) + " " + str(D_in) + " " +str(H) + " " + str(D_out) + "\n" 

    lr1 = np.random.uniform()
    momentum1 = np.random.uniform()
    weight_decay1 = np.random.uniform()
    arg5 = str(lr1) + " " + str(momentum1) + " " +str(weight_decay1) + "\n"

    lr2 = np.random.uniform()
    momentum2 = np.random.uniform()
    weight_decay2 = np.random.uniform()
    arg6 = str(lr2) + " " + str(momentum2) + " " +str(weight_decay2) + "\n"

    Conv2d01 = np.random.randint(1, 101)
    Conv2d11 = np.random.randint(1, 101)
    Conv2d21 = np.random.randint(1, 101)
    arg7 = str(Conv2d01) + " " + str(Conv2d11) + " " +str(Conv2d21) + "\n"

    Conv2d02 = np.random.randint(1, 101)
    Conv2d12 = np.random.randint(1, 101)
    Conv2d22 = np.random.randint(1, 101)
    arg8 = str(Conv2d02) + " " + str(Conv2d12) + " " +str(Conv2d22) + "\n"

    rnd01 = np.random.randint(1, 11)
    rnd02 = np.random.randint(1, 11)
    rnd03 = np.random.randint(1, 11)
    rnd04 = np.random.randint(1, 11)
    arg9 = str(rnd01) + " " + str(rnd02) + " " +str(rnd03) + " " +str(rnd04) + "\n"

    rnd11 = np.random.randint(1, 11)
    rnd12 = np.random.randint(1, 11)
    rnd13 = np.random.randint(1, 11)
    rnd14 = np.random.randint(1, 11)
    arg10 = str(rnd11) + " " + str(rnd12) + " " +str(rnd13) + " " +str(rnd14) + "\n"

    rnd21 = np.random.randint(1, 11)
    rnd22 = np.random.randint(1, 11)
    rnd23 = np.random.randint(1, 11)
    arg11 = str(rnd21) + " " + str(rnd22) + " " +str(rnd23) + "\n"

    rnd31 = np.random.randint(1, 11)
    rnd32 = np.random.randint(1, 11)
    rnd33 = np.random.randint(1, 11)
    arg12 = str(rnd31) + " " + str(rnd32) + " " +str(rnd33) + "\n"

    rnd41 = np.random.randint(1, 101)
    rnd42 = np.random.randint(1, 101)
    rnd43 = np.random.randint(1, 101)
    rnd44 = np.random.randint(1, 101)
    rnd45 = np.random.randint(1, 101)
    rnd46 = np.random.randint(1, 101)
    rnd47 = np.random.randint(1, 101)
    rnd48 = np.random.randint(3, 8)
    arg13 = str(rnd41) + " " + str(rnd42) + " " +str(rnd43) + " " +str(rnd44) + " " + str(rnd45) + " " + str(rnd46) + " " +str(rnd47) + " " +str(rnd48) + "\n"

    lines = [str(torch_seed)+'\n', str(numpy_seed)+'\n', str(random_int1)+'\n', str(random_int2)+'\n', arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13]
    logfile_name = "./logs/input" + id + ".log" 

    with open(logfile_name, "w") as f:
        f.writelines(lines)
        f.close()

    with open("./conf.txt", "w") as f:
        f.writelines(lines)
        f.close()


if __name__ == "__main__":
    if sys.argv[1]:
        generate_inputs(sys.argv[1])

    # with open("./conf.txt", "r") as f:
    # 	lll = f.readlines()
    # 	f.close()

    # # for i in lll:
    # # 	print(i)
    # print(lll[-1].split(" ")[7][:-2])
    # print(lll[-2].split(" "))
    # print(lll[-3].split(" "))
    # print(lll[-4].split(" "))
    # print(lll[-5].split(" "))
