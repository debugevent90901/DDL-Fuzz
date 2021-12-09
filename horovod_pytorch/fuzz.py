

import time, hashlib
import numpy as np
from torch.nn.functional import log_softmax

# range of the random seed, int 16
LOWER_BOUND = 0
UPPER_BOUND = 65536

class Fuzzer():
    def __init__(self):
        seed = str(np.random.randint(LOWER_BOUND, UPPER_BOUND))
        self.id = hashlib.sha1(seed.encode("utf-8")).hexdigest()
        self.logfile = "./log_" + id[:5]

        with open(self.logfile, "w") as log:
            log.write("Test " + self.id + " init.\n\n\n")
            log.close()

    def log_torch_seed(self, test_method_name):
        seed = str(np.random.randint(LOWER_BOUND, UPPER_BOUND))
        with open(self.logfile, "a") as log:
            log.write("Set pytorch.manual_seed for " + test_method_name + " as: " + str(seed) + "\n")
            log.close()
        
        return seed
    
    def log_torch_seed(self, test_method_name):
        seed = str(np.random.randint(LOWER_BOUND, UPPER_BOUND))
        with open(self.logfile, "a") as log:
            log.write("Set numpy.seed for " + test_method_name + " as: " + str(seed) + "\n")
            log.close()
        
        return seed
    
    def log_tensor_shape(self, test_method_name):
        t = str(np.random.randint(5, 25))
        with open(self.logfile, "a") as log:
            log.write("Set tensor for " + test_method_name + " ([t] * dim), where t=" + t + "\n")
            log.close()
        
        return t




f = Fuzzer()
a = f.log_torch_seed(".test_allreduce")