# IMPORT operations, should not be mutated
import os, sys, time, argparse, traceback
from distutils.version import LooseVersion
from filelock import FileLock

import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data.distributed
import horovod.torch as hvd

# ARGUMENTS, should not be mutated
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
parser.add_argument('--gradient-predivide-factor', type=float, default=1.0, help='apply gradient predivide factor in optimizer (default: 1.0)')

# MODEL, should not be mutated
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

# ENTRY
if __name__ == '__main__':
    try: 
        args = parser.parse_args()
        args.cuda = torch.cuda.is_available()

        # Horovod: initialize library.
        hvd.init()
        torch.manual_seed(args.seed)

        if args.cuda:
            # Horovod: pin GPU to local rank.
            torch.cuda.set_device(hvd.local_rank())
            torch.cuda.manual_seed(args.seed)

        # Horovod: limit # of CPU threads to be used per worker.
        torch.set_num_threads(1)

        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent issues with Infiniband implementations that are not fork-safe
        if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
            kwargs['multiprocessing_context'] = 'forkserver'

        data_dir = './data'
        with FileLock(os.path.expanduser("~/.horovod_lock")):
            train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))

        # Horovod: use DistributedSampler to partition the training data.
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)

        model = Net()
        lr_scaler = hvd.size()

        if args.cuda:
            # Move model to GPU.
            model.cuda()

        # Horovod: scale learning rate by lr_scaler.
        optimizer = optim.SGD(model.parameters(), lr=args.lr * lr_scaler, momentum=args.momentum)
        # Horovod: wrap optimizer with DistributedOptimizer.
        optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters(), compression=hvd.Compression.none, op=hvd.Average, gradient_predivide_factor=args.gradient_predivide_factor)

        # Horovod: broadcast parameters & optimizer state.
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        for epoch in range(1, args.epochs + 1):
            # Horovod: set epoch to sampler for shuffling.
            train_sampler.set_epoch(epoch)
            for batch_idx, (data, target) in enumerate(train_loader):
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                if batch_idx % args.log_interval == 0:
                    # Horovod: use train_sampler to determine the number of examples in this worker's partition.
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_sampler), 100. * batch_idx / len(train_loader), loss.item()))

    except:
        err = open(sys.argv[0][:-3] + ".log", 'w')
        err.write(traceback.format_exc())
        err.close()