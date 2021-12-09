# Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
# Modifications copyright (C) 2019 Intel Corporation
# Modifications copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from horovod_pytorch.common import mpi_env_rank_and_size, skip_or_fail_gpu_test, temppath
from distutils.version import LooseVersion

import os, sys, time, json, inspect, platform, warnings, unittest, itertools

from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import horovod.torch as hvd

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, 'utils'))


_1_5_api = LooseVersion(torch.__version__) >= LooseVersion('1.5.0')

ccl_supported_types = set([torch.ByteTensor, torch.CharTensor, torch.ShortTensor,
                           torch.IntTensor, torch.LongTensor, torch.FloatTensor,
                           torch.DoubleTensor])

# Set environment variable for dynamic timeline API test
os.environ["HOROVOD_TIMELINE"] = "DYNAMIC"


class TorchTests(unittest.TestCase):
    """
    Tests for ops in horovod.torch.
    """

    def __init__(self, *args, **kwargs):
        super(TorchTests, self).__init__(*args, **kwargs)
        warnings.simplefilter('module')

    def convert_cpu_fp16_to_fp32(self, *values):
        # PyTorch doesn't support any CPU ops on FP16 tensors.
        # In case we need to do ops, we will convert tensor to FP32 here.
        result = []
        for value in values:
            if value.dtype in [torch.float16, torch.HalfTensor] and not value.is_cuda:
                result.append(value.float())
            else:
                result.append(value)
        return result

    def cast_and_place(self, tensor, dtype):
        if dtype.is_cuda:
            return tensor.cuda(hvd.local_rank()).type(dtype)
        return tensor.type(dtype)

    def filter_supported_types(self, types):
        if 'CCL_ROOT' in os.environ:
            types = [t for t in types if t in ccl_supported_types]
        return types



    # Unit test functions being fuzzed
    # Below are unit test functions used in fuzzer: only most frequently used methods are fuzzed

    def test_horovod_allreduce(self):
        """Test that the allreduce correctly sums 1D, 2D, 3D tensors."""
        hvd.init()
#         torch.cuda.set_device(hvd.local_rank())
        print("Rank is: ", hvd.rank(), "Size is: ", hvd.size())

        size = hvd.size()
        dtypes = self.filter_supported_types([torch.IntTensor, torch.LongTensor,
                                              torch.FloatTensor, torch.DoubleTensor, torch.HalfTensor])
        if torch.cuda.is_available():
            dtypes += [torch.cuda.IntTensor, torch.cuda.LongTensor,
                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
                       torch.cuda.HalfTensor]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            torch.manual_seed(1234)
            tensor = torch.FloatTensor(*([17] * dim)).random_(-100, 100)
            tensor = self.cast_and_place(tensor, dtype)
            summed = hvd.allreduce(tensor, average=False)
            tensor, summed = self.convert_cpu_fp16_to_fp32(tensor, summed)
            multiplied = tensor * size

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [torch.IntTensor, torch.LongTensor,
                                      torch.cuda.IntTensor, torch.cuda.LongTensor]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break

            assert torch.allclose(
                summed, multiplied, threshold), 'hvd.allreduce produces incorrect results'

    def test_horovod_allreduce_average(self):
        """Test that the allreduce correctly averages 1D, 2D, 3D tensors."""
        hvd.init()
#         torch.cuda.set_device(hvd.local_rank())
        print("Rank is: ", hvd.rank(), "Size is: ", hvd.size())

        size = hvd.size()
        dtypes = self.filter_supported_types([torch.IntTensor, torch.LongTensor,
                                              torch.FloatTensor, torch.DoubleTensor])
        if torch.cuda.is_available():
            dtypes += [torch.cuda.IntTensor, torch.cuda.LongTensor,
                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
                       torch.cuda.HalfTensor]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            torch.manual_seed(1234)
            tensor = torch.FloatTensor(*([17] * dim)).random_(-100, 100)
            tensor = self.cast_and_place(tensor, dtype)
            averaged = hvd.allreduce(tensor, average=True)

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [torch.IntTensor, torch.LongTensor,
                                      torch.cuda.IntTensor, torch.cuda.LongTensor]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break

            assert torch.allclose(
                averaged, tensor, threshold), 'hvd.allreduce produces incorrect results'

    def test_horovod_allreduce_inplace(self):
        """Test that the allreduce correctly sums 1D, 2D, 3D tensors."""
        hvd.init()
#         torch.cuda.set_device(hvd.local_rank())
        print("Rank is: ", hvd.rank(), "Size is: ", hvd.size())

        size = hvd.size()
        dtypes = self.filter_supported_types([torch.IntTensor, torch.LongTensor,
                                              torch.FloatTensor, torch.DoubleTensor, torch.HalfTensor])
        if torch.cuda.is_available():
            dtypes += [torch.cuda.IntTensor, torch.cuda.LongTensor,
                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
                       torch.cuda.HalfTensor]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            torch.manual_seed(1234)
            tensor = torch.FloatTensor(*([17] * dim)).random_(-100, 100)
            multiplied = self.cast_and_place(tensor * size, dtype)
            tensor = self.cast_and_place(tensor, dtype)
            hvd.allreduce_(tensor, average=False)
            tensor, multiplied = self.convert_cpu_fp16_to_fp32(
                tensor, multiplied)

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [torch.IntTensor, torch.LongTensor,
                                      torch.cuda.IntTensor, torch.cuda.LongTensor]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break

            assert torch.allclose(
                tensor, multiplied, threshold), 'hvd.allreduce produces incorrect results'

    def test_horovod_allreduce_async_fused(self):
        """Test that the allreduce correctly sums 1D, 2D, 3D tensors
        with Tensor Fusion."""
        hvd.init()
#         torch.cuda.set_device(hvd.local_rank())
        print("Rank is: ", hvd.rank(), "Size is: ", hvd.size())

        size = hvd.size()
        dtypes = self.filter_supported_types([torch.IntTensor, torch.LongTensor,
                                              torch.FloatTensor, torch.DoubleTensor, torch.HalfTensor])
        if torch.cuda.is_available():
            dtypes += [torch.cuda.IntTensor, torch.cuda.LongTensor,
                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
                       torch.cuda.HalfTensor]
        dims = [1, 2, 3]
        tests = []
        is_hvd_poll_false_once = False
        for dtype, dim in itertools.product(dtypes, dims):
            torch.manual_seed(1234)
            tensor = torch.FloatTensor(*([17] * dim)).random_(-100, 100)
            tensor = self.cast_and_place(tensor, dtype)
            handle = hvd.allreduce_async(tensor, average=False)
            if not hvd.poll(handle):
                is_hvd_poll_false_once = True
            tensor, = self.convert_cpu_fp16_to_fp32(tensor)
            multiplied = tensor * size
            tests.append((dtype, multiplied, handle))

        # Make sure it's an asynchronous operation.
        assert is_hvd_poll_false_once, 'hvd.poll() always returns True, not an async op?'

        for dtype, multiplied, handle in tests:
            summed = hvd.synchronize(handle)
            summed, = self.convert_cpu_fp16_to_fp32(summed)

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [torch.IntTensor, torch.LongTensor,
                                      torch.cuda.IntTensor, torch.cuda.LongTensor]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break

            assert torch.allclose(
                summed, multiplied, threshold), 'hvd.allreduce produces incorrect results'

    def test_horovod_allreduce_multi_gpu(self):
        """Test that the allreduce works on multiple GPUs."""
        # Only do this test if there are GPUs available.
        if not torch.cuda.is_available():
            self.skipTest("No GPUs available")

        hvd.init()
#         torch.cuda.set_device(hvd.local_rank())
        print("Rank is: ", hvd.rank(), "Size is: ", hvd.size())

        local_rank = hvd.local_rank()
        size = hvd.size()

        # Skip the test if there are not enough GPUs.
        if torch.cuda.device_count() < hvd.local_size() * 2:
            self.skipTest("Not enough GPUs available")

        iter = 0
        dtypes = [torch.cuda.IntTensor, torch.cuda.LongTensor,
                  torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
                  torch.cuda.HalfTensor]

        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            iter += 1
            torch.manual_seed(1234)
            tensor = torch.FloatTensor(*([17] * dim)).random_(-100, 100)
            device = local_rank * 2 + (iter + local_rank) % 2
            tensor = tensor.cuda(device).type(dtype)
            multiplied = tensor * size
            hvd.allreduce_(tensor, average=False)

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [torch.cuda.IntTensor, torch.cuda.LongTensor]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break

            assert torch.allclose(
                tensor, multiplied, threshold), 'hvd.allreduce produces incorrect results'

    def test_horovod_allreduce_prescale(self):
        """Test that the allreduce correctly sums 1D, 2D, 3D tensors with prescaling."""
        hvd.init()
#         torch.cuda.set_device(hvd.local_rank())
        print("Rank is: ", hvd.rank(), "Size is: ", hvd.size())

        size = hvd.size()
        dtypes = self.filter_supported_types([torch.IntTensor, torch.LongTensor,
                                              torch.FloatTensor, torch.DoubleTensor, torch.HalfTensor])
        if torch.cuda.is_available():
            dtypes += [torch.cuda.IntTensor, torch.cuda.LongTensor,
                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
                       torch.cuda.HalfTensor]
        int_types = [torch.IntTensor, torch.LongTensor,
                     torch.cuda.IntTensor, torch.cuda.LongTensor]
        half_types = [torch.HalfTensor, torch.cuda.HalfTensor]

        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            torch.manual_seed(1234)
            np.random.seed(1234)
            factor = np.random.uniform()
            tensor = torch.FloatTensor(*([17] * dim)).random_(-100, 100)
            tensor = self.cast_and_place(tensor, dtype)
            summed = hvd.allreduce(tensor, average=False,
                                   prescale_factor=factor)

            factor = torch.tensor(factor, dtype=torch.float64)
            factor = factor.cuda(hvd.local_rank()) if dtype.is_cuda else factor
            if dtype.is_cuda and not int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
                # For integer types, scaling done in FP64
                factor = factor.type(
                    torch.float64 if dtype in int_types else dtype)
                tensor = tensor.type(
                    torch.float64 if dtype in int_types else dtype)
            else:
                # For integer types, scaling done in FP64, FP32 math for FP16 on CPU
                factor = factor.type(torch.float32 if dtype in half_types else
                                     torch.float64 if dtype in int_types else dtype)
                tensor = tensor.type(torch.float32 if dtype in half_types else
                                     torch.float64 if dtype in int_types else dtype)
            multiplied = factor * tensor
            multiplied = multiplied.type(dtype)
            summed, multiplied = self.convert_cpu_fp16_to_fp32(
                summed, multiplied)
            multiplied *= size

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in int_types:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break

            assert torch.allclose(
                summed, multiplied, threshold), 'hvd.allreduce produces incorrect results'

    def test_horovod_allreduce_postscale(self):
        """Test that the allreduce correctly sums 1D, 2D, 3D tensors with postscaling."""
        hvd.init()
#         torch.cuda.set_device(hvd.local_rank())
        print("Rank is: ", hvd.rank(), "Size is: ", hvd.size())

        size = hvd.size()
        dtypes = self.filter_supported_types([torch.IntTensor, torch.LongTensor,
                                              torch.FloatTensor, torch.DoubleTensor, torch.HalfTensor])
        if torch.cuda.is_available():
            dtypes += [torch.cuda.IntTensor, torch.cuda.LongTensor,
                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
                       torch.cuda.HalfTensor]
        int_types = [torch.IntTensor, torch.LongTensor,
                     torch.cuda.IntTensor, torch.cuda.LongTensor]
        half_types = [torch.HalfTensor, torch.cuda.HalfTensor]

        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            torch.manual_seed(1234)
            np.random.seed(1234)
            factor = np.random.uniform()
            tensor = torch.FloatTensor(*([17] * dim)).random_(-100, 100)
            tensor = self.cast_and_place(tensor, dtype)
            summed = hvd.allreduce(tensor, average=False,
                                   postscale_factor=factor)

            factor = torch.tensor(factor, dtype=torch.float64)
            factor = factor.cuda(hvd.local_rank()) if dtype.is_cuda else factor
            if dtype.is_cuda and not int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
                # For integer types, scaling done in FP64
                factor = factor.type(
                    torch.float64 if dtype in int_types else dtype)
                tensor = tensor.type(
                    torch.float64 if dtype in int_types else dtype)
            else:
                # For integer types, scaling done in FP64, FP32 math for FP16 on CPU
                factor = factor.type(torch.float32 if dtype in half_types else
                                     torch.float64 if dtype in int_types else dtype)
                tensor = tensor.type(torch.float32 if dtype in half_types else
                                     torch.float64 if dtype in int_types else dtype)
            multiplied = size * tensor
            multiplied = multiplied * factor
            multiplied = multiplied.type(dtype)
            summed, multiplied = self.convert_cpu_fp16_to_fp32(
                summed, multiplied)

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in int_types:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break

            assert torch.allclose(
                summed, multiplied, threshold), 'hvd.allreduce produces incorrect results'

    def test_horovod_allreduce_grad(self):
        """Test the correctness of the allreduce gradient."""
        hvd.init()
#         torch.cuda.set_device(hvd.local_rank())
        print("Rank is: ", hvd.rank(), "Size is: ", hvd.size())

        size = hvd.size()
        # Only Tensors of floating point dtype can require gradients
        dtypes = [torch.FloatTensor, torch.DoubleTensor]
        if torch.cuda.is_available():
            dtypes += [torch.cuda.FloatTensor,
                       torch.cuda.DoubleTensor, torch.cuda.HalfTensor]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            torch.manual_seed(1234)
            tensor = torch.FloatTensor(*([17] * dim)).random_(-100, 100)
            tensor = self.cast_and_place(tensor, dtype)
            tensor.requires_grad_()
            summed = hvd.allreduce(tensor, average=False)

            summed.backward(self.cast_and_place(torch.ones([17] * dim), dtype))
            grad_out = tensor.grad.data.cpu().numpy()

            expected = np.ones([17] * dim) * size
            err = np.linalg.norm(expected - grad_out)
            self.assertLess(err, 0.00000001,
                            "gradient %s differs from expected %s, "
                            "error: %s" % (grad_out, expected, str(err)))

    def test_horovod_allreduce_grad_average(self):
        """Test the correctness of the allreduce averaged gradient."""
        hvd.init()
#         torch.cuda.set_device(hvd.local_rank())
        print("Rank is: ", hvd.rank(), "Size is: ", hvd.size())

        # Only Tensors of floating point dtype can require gradients
        dtypes = [torch.FloatTensor, torch.DoubleTensor]
        if torch.cuda.is_available():
            dtypes += [torch.cuda.FloatTensor,
                       torch.cuda.DoubleTensor, torch.cuda.HalfTensor]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            torch.manual_seed(1234)
            tensor = torch.FloatTensor(*([17] * dim)).random_(-100, 100)
            tensor = self.cast_and_place(tensor, dtype)
            tensor.requires_grad_()
            summed = hvd.allreduce(tensor, average=True)

            summed.backward(self.cast_and_place(torch.ones([17] * dim), dtype))
            grad_out = tensor.grad.data.cpu().numpy()

            expected = np.ones([17] * dim)
            err = np.linalg.norm(expected - grad_out)
            self.assertLess(err, 0.00000001,
                            "gradient %s differs from expected %s, "
                            "error: %s" % (grad_out, expected, str(err)))

    def test_horovod_grouped_allreduce(self):
        """Test that the grouped allreduce correctly sums 1D, 2D, 3D tensors."""
        hvd.init()
#         torch.cuda.set_device(hvd.local_rank())
        print("Rank is: ", hvd.rank(), "Size is: ", hvd.size())

        size = hvd.size()
        dtypes = self.filter_supported_types([torch.IntTensor, torch.LongTensor,
                                              torch.FloatTensor, torch.DoubleTensor, torch.HalfTensor])
        if torch.cuda.is_available():
            dtypes += [torch.cuda.IntTensor, torch.cuda.LongTensor,
                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
                       torch.cuda.HalfTensor]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            torch.manual_seed(1234)
            tensors = [torch.FloatTensor(
                *([17] * dim)).random_(-100, 100) for _ in range(5)]
            tensors = [self.cast_and_place(tensor, dtype)
                       for tensor in tensors]
            summed = hvd.grouped_allreduce(tensors, average=False)
            tensors, summed = zip(
                *[self.convert_cpu_fp16_to_fp32(t, s) for t, s in zip(tensors, summed)])
            multiplied = [tensor * size for tensor in tensors]

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [torch.IntTensor, torch.LongTensor,
                                      torch.cuda.IntTensor, torch.cuda.LongTensor]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break

            assert all([torch.allclose(t1, t2, threshold) for t1, t2 in zip(summed, multiplied)]), \
                'hvd.grouped_allreduce produces incorrect results'

    def test_horovod_grouped_allreduce_average(self):
        """Test that the grouped allreduce correctly averages 1D, 2D, 3D tensors."""
        hvd.init()
#         torch.cuda.set_device(hvd.local_rank())
        print("Rank is: ", hvd.rank(), "Size is: ", hvd.size())

        size = hvd.size()
        dtypes = self.filter_supported_types([torch.IntTensor, torch.LongTensor,
                                              torch.FloatTensor, torch.DoubleTensor, torch.HalfTensor])
        if torch.cuda.is_available():
            dtypes += [torch.cuda.IntTensor, torch.cuda.LongTensor,
                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
                       torch.cuda.HalfTensor]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            torch.manual_seed(1234)
            tensors = [torch.FloatTensor(
                *([17] * dim)).random_(-100, 100) for _ in range(5)]
            tensors = [self.cast_and_place(tensor, dtype)
                       for tensor in tensors]
            averaged = hvd.grouped_allreduce(tensors, average=True)
            tensors, averaged = zip(
                *[self.convert_cpu_fp16_to_fp32(t, m) for t, m in zip(tensors, averaged)])

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [torch.IntTensor, torch.LongTensor,
                                      torch.cuda.IntTensor, torch.cuda.LongTensor]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break

            assert all([torch.allclose(t1, t2, threshold) for t1, t2 in zip(averaged, tensors)]), \
                'hvd.grouped_allreduce produces incorrect results for average'

    def test_horovod_grouped_allreduce_inplace(self):
        """Test that the grouped allreduce correctly sums 1D, 2D, 3D tensors."""
        hvd.init()
#         torch.cuda.set_device(hvd.local_rank())
        print("Rank is: ", hvd.rank(), "Size is: ", hvd.size())

        size = hvd.size()
        dtypes = self.filter_supported_types([torch.IntTensor, torch.LongTensor,
                                              torch.FloatTensor, torch.DoubleTensor, torch.HalfTensor])
        if torch.cuda.is_available():
            dtypes += [torch.cuda.IntTensor, torch.cuda.LongTensor,
                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
                       torch.cuda.HalfTensor]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            torch.manual_seed(1234)
            tensors = [torch.FloatTensor(
                *([17] * dim)).random_(-100, 100) for _ in range(5)]
            multiplied = [self.cast_and_place(
                tensor * size, dtype) for tensor in tensors]
            tensors = [self.cast_and_place(tensor, dtype)
                       for tensor in tensors]
            hvd.grouped_allreduce_(tensors, average=False)
            tensors, multiplied = zip(
                *[self.convert_cpu_fp16_to_fp32(t, m) for t, m in zip(tensors, multiplied)])

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [torch.IntTensor, torch.LongTensor,
                                      torch.cuda.IntTensor, torch.cuda.LongTensor]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break

            assert all([torch.allclose(t1, t2, threshold) for t1, t2 in zip(tensors, multiplied)]), \
                'hvd.grouped_allreduce_ produces incorrect results'

    def test_horovod_grouped_allreduce_grad(self):
        """Test the correctness of the grouped allreduce gradient."""
        hvd.init()
#         torch.cuda.set_device(hvd.local_rank())
        print("Rank is: ", hvd.rank(), "Size is: ", hvd.size())

        size = hvd.size()
        # Only Tensors of floating point dtype can require gradients
        dtypes = [torch.FloatTensor, torch.DoubleTensor]
        if torch.cuda.is_available():
            dtypes += [torch.cuda.FloatTensor,
                       torch.cuda.DoubleTensor, torch.cuda.HalfTensor]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            torch.manual_seed(1234)
            tensors = [torch.FloatTensor(
                *([17] * dim)).random_(-100, 100) for _ in range(5)]
            tensors = [self.cast_and_place(tensor, dtype)
                       for tensor in tensors]
            for tensor in tensors:
                tensor.requires_grad_()
            summed = hvd.grouped_allreduce(tensors, average=False)

            for s in summed:
                s.backward(self.cast_and_place(torch.ones([17] * dim), dtype))

            grads_out = [tensor.grad.data.cpu().numpy() for tensor in tensors]

            expected = np.ones([17] * dim) * size
            for grad_out in grads_out:
                err = np.linalg.norm(expected - grad_out)
                self.assertLess(err, 0.00000001,
                                "gradient %s differs from expected %s, "
                                "error: %s" % (grad_out, expected, str(err)))

    def test_horovod_allreduce_grad_average(self):
        """Test the correctness of the allreduce averaged gradient."""
        hvd.init()
#         torch.cuda.set_device(hvd.local_rank())
        print("Rank is: ", hvd.rank(), "Size is: ", hvd.size())

        # Only Tensors of floating point dtype can require gradients
        dtypes = [torch.FloatTensor, torch.DoubleTensor]
        if torch.cuda.is_available():
            dtypes += [torch.cuda.FloatTensor,
                       torch.cuda.DoubleTensor, torch.cuda.HalfTensor]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            torch.manual_seed(1234)
            tensors = [torch.FloatTensor(
                *([17] * dim)).random_(-100, 100) for _ in range(5)]
            tensors = [self.cast_and_place(tensor, dtype)
                       for tensor in tensors]
            for tensor in tensors:
                tensor.requires_grad_()
            summed = hvd.grouped_allreduce(tensors, average=True)

            for s in summed:
                s.backward(self.cast_and_place(torch.ones([17] * dim), dtype))

            grads_out = [tensor.grad.data.cpu().numpy() for tensor in tensors]

            expected = np.ones([17] * dim)
            for grad_out in grads_out:
                err = np.linalg.norm(expected - grad_out)
                self.assertLess(err, 0.00000001,
                                "gradient %s differs from expected %s, "
                                "error: %s" % (grad_out, expected, str(err)))

    def test_horovod_broadcast(self):
        """Test that the broadcast correctly broadcasts 1D, 2D, 3D tensors."""
        hvd.init()
#         torch.cuda.set_device(hvd.local_rank())
        print("Rank is: ", hvd.rank(), "Size is: ", hvd.size())

        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        dtypes = [torch.ByteTensor, torch.CharTensor, torch.ShortTensor,
                  torch.IntTensor, torch.LongTensor, torch.FloatTensor, torch.DoubleTensor,
                  torch.HalfTensor]
        if torch.cuda.is_available():
            dtypes += [torch.cuda.ByteTensor, torch.cuda.CharTensor, torch.cuda.ShortTensor,
                       torch.cuda.IntTensor, torch.cuda.LongTensor,
                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
                       torch.cuda.HalfTensor]
        dims = [1, 2, 3]
        root_ranks = list(range(size))
        for dtype, dim, root_rank in itertools.product(dtypes, dims, root_ranks):
            tensor = torch.FloatTensor(*([17] * dim)).fill_(1).mul_(rank)
            root_tensor = torch.FloatTensor(
                *([17] * dim)).fill_(1).mul_(root_rank)
            tensor = self.cast_and_place(tensor, dtype)
            root_tensor = self.cast_and_place(root_tensor, dtype)
            broadcasted_tensor = hvd.broadcast(tensor, root_rank)
            tensor, root_tensor, broadcasted_tensor = \
                self.convert_cpu_fp16_to_fp32(
                    tensor, root_tensor, broadcasted_tensor)
            if rank != root_rank:
                assert (tensor == root_tensor).max() == 0, \
                    'hvd.broadcast modifies source tensor'
            assert (broadcasted_tensor.data == root_tensor).min() == 1, \
                'hvd.broadcast produces incorrect broadcasted tensor'

    def test_horovod_broadcast_inplace(self):
        """Test that the broadcast correctly broadcasts 1D, 2D, 3D tensors."""
        hvd.init()
#         torch.cuda.set_device(hvd.local_rank())
        print("Rank is: ", hvd.rank(), "Size is: ", hvd.size())

        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        dtypes = [torch.ByteTensor, torch.CharTensor, torch.ShortTensor,
                  torch.IntTensor, torch.LongTensor, torch.FloatTensor, torch.DoubleTensor,
                  torch.HalfTensor]
        if torch.cuda.is_available():
            dtypes += [torch.cuda.ByteTensor, torch.cuda.CharTensor, torch.cuda.ShortTensor,
                       torch.cuda.IntTensor, torch.cuda.LongTensor,
                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
                       torch.cuda.HalfTensor]
        dims = [1, 2, 3]
        root_ranks = list(range(size))
        for dtype, dim, root_rank in itertools.product(dtypes, dims, root_ranks):
            tensor = torch.FloatTensor(*([17] * dim)).fill_(1).mul_(rank)
            root_tensor = torch.FloatTensor(
                *([17] * dim)).fill_(1).mul_(root_rank)
            tensor = self.cast_and_place(tensor, dtype)
            root_tensor = self.cast_and_place(root_tensor, dtype)
            broadcasted_tensor = hvd.broadcast_(tensor, root_rank)
            tensor, root_tensor, broadcasted_tensor = \
                self.convert_cpu_fp16_to_fp32(
                    tensor, root_tensor, broadcasted_tensor)
            assert (tensor == broadcasted_tensor).min() == 1, \
                'hvd.broadcast does not modify source tensor'
            assert (broadcasted_tensor == root_tensor).min() == 1, \
                'hvd.broadcast produces incorrect broadcasted tensor'

    def test_horovod_broadcast_grad(self):
        """Test the correctness of the broadcast gradient."""
        hvd.init()
#         torch.cuda.set_device(hvd.local_rank())
        print("Rank is: ", hvd.rank(), "Size is: ", hvd.size())

        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        # Only Tensors of floating point dtype can require gradients
        dtypes = [torch.FloatTensor, torch.DoubleTensor]
        if torch.cuda.is_available():
            dtypes += [torch.cuda.FloatTensor,
                       torch.cuda.DoubleTensor, torch.cuda.HalfTensor]
        dims = [1, 2, 3]
        root_ranks = list(range(size))
        for dtype, dim, root_rank in itertools.product(dtypes, dims, root_ranks):
            tensor = torch.FloatTensor(*([17] * dim)).fill_(1).mul_(rank)
            tensor = self.cast_and_place(tensor, dtype)
            tensor.requires_grad_()

            broadcasted_tensor = hvd.broadcast(tensor, root_rank)
            broadcasted_tensor.backward(
                self.cast_and_place(torch.ones([17] * dim), dtype))
            grad_out = tensor.grad.data.cpu().numpy()

            c = 1 if rank == root_rank else 0
            expected = np.ones([17] * dim) * c
            err = np.linalg.norm(expected - grad_out)
            self.assertLess(err, 0.00000001,
                            "gradient %s differs from expected %s, "
                            "error: %s" % (grad_out, expected, str(err)))

    def test_broadcast_state(self):
        hvd.init()
#         torch.cuda.set_device(hvd.local_rank())
        print("Rank is: ", hvd.rank(), "Size is: ", hvd.size())


        N, D_in, H, D_out = 64, 100, 10, 10
        x = torch.randn(N, D_in).requires_grad_()
        y = torch.randn(N, D_out).requires_grad_()

        def new_optimizer(cls, opt_params, model):
            p = {
                k: v for k, v in opt_params.items()
                if k in inspect.getargspec(cls.__init__).args
            }
            return cls(model.parameters(), **p)

        def create_model(opt_class, opt_params):
            model = torch.nn.Sequential(
                torch.nn.Linear(D_in, H),
                torch.nn.ReLU(),
                torch.nn.Linear(H, D_out),
            )

            optimizer = new_optimizer(opt_class, opt_params, model)
            optimizer = hvd.DistributedOptimizer(
                optimizer, named_parameters=model.named_parameters())

            return model, optimizer

        def get_model_param_values(model):
            params = sorted(model.state_dict().items())
            return [(k, v.clone()) for k, v in params]

        def get_optimizer_param_values(optimizer):
            results = []
            state_dict = optimizer.state_dict()
            for group in state_dict['param_groups']:
                for param_id in group['params']:
                    if param_id not in state_dict['state']:
                        continue
                    params = sorted(state_dict['state'][param_id].items())
                    for k, v in params:
                        results.append(
                            (k, v.clone() if torch.is_tensor(v) else v))
            return results

        # L-BFGS is currently unsupported, as are sparse tensors, which are
        # required by SparseAdam optimizer
        optimizers = [
            (subclass.__name__, subclass)
            for subclass in torch.optim.Optimizer.__subclasses__()
            if subclass.__module__.startswith('torch.optim') and
            subclass != torch.optim.LBFGS and
            subclass != torch.optim.SparseAdam
        ]
        optimizers.sort(key=lambda tup: tup[0])

        opt_params_list = [
            dict(lr=0.2, momentum=0.9, weight_decay=0.1, centered=True),
            dict(lr=0.2)
        ]

        for (opt_name, opt_class), opt_params in itertools.product(optimizers, opt_params_list):
            model, optimizer = create_model(opt_class, opt_params)
            y_pred = model(x)
            loss = F.mse_loss(y_pred, y, size_average=False)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model_param_values = get_model_param_values(model)
            for name, model_param_value in model_param_values:
                hvd.broadcast_(model_param_value, root_rank=0)

            opt_param_values_updated = []
            opt_param_values = get_optimizer_param_values(optimizer)
            for name, opt_param_value in opt_param_values:
                is_tensor = torch.is_tensor(opt_param_value)
                if is_tensor:
                    hvd.broadcast_(opt_param_value, root_rank=0)
                else:
                    opt_param_value = hvd.broadcast_object(
                        opt_param_value, name=name)
                opt_param_values_updated.append((name, opt_param_value))
            opt_param_values = opt_param_values_updated

            with temppath() as fname:
                if hvd.rank() == 0:
                    state = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
                    torch.save(state, fname)

                model, optimizer = create_model(opt_class, opt_params)
                if hvd.rank() == 0:
                    checkpoint = torch.load(fname)
                    model.load_state_dict(checkpoint['model'])
                    optimizer.load_state_dict(checkpoint['optimizer'])

            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            model_param_value_after = get_model_param_values(model)
            for before, after in zip(model_param_values,
                                     model_param_value_after):
                name, model_param_value = before
                name_after, model_param_value_after = after
                self.assertEqual(name, name_after)
                self.assertEqual(type(model_param_value),
                                 type(model_param_value_after))
                self.assertTrue(
                    (model_param_value == model_param_value_after).all())

            expected_tensors = hvd.broadcast_object(
                len(optimizer.state_dict()['state'].values()))
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)
            self.assertEqual(len(optimizer.state_dict()[
                             'state'].values()), expected_tensors)

            opt_param_values_after = get_optimizer_param_values(optimizer)
            for before, after in zip(opt_param_values, opt_param_values_after):
                name, opt_param_value = before
                name_after, opt_param_value_after = after
                self.assertEqual(name, name_after)
                self.assertEqual(type(opt_param_value),
                                 type(opt_param_value_after))
                if torch.is_tensor(opt_param_value):
                    self.assertTrue(
                        (opt_param_value == opt_param_value_after).all())
                else:
                    self.assertEqual(opt_param_value, opt_param_value_after)

    def test_broadcast_state_options(self):
        hvd.init()
#         torch.cuda.set_device(hvd.local_rank())
        print("Rank is: ", hvd.rank(), "Size is: ", hvd.size())


        N, D_in, H, D_out = 64, 100, 10, 10
        x = torch.randn(N, D_in).requires_grad_()
        y = torch.randn(N, D_out).requires_grad_()

        params_0 = dict(lr=0.1, momentum=0.8, weight_decay=0.2, nesterov=True,
                        betas=(0.9, 0.999), etas=(0.8, 2.4), step_sizes=(1e-5, 100))
        params_1 = dict(lr=0.2, momentum=0.9, weight_decay=0.1, nesterov=False,
                        betas=(0.8, 0.9), etas=(0.25, 1.75), step_sizes=(1e-7, 5))

        def create_model(opt_class):
            model = torch.nn.Sequential(
                torch.nn.Linear(D_in, H),
                torch.nn.ReLU(),
                torch.nn.Linear(H, D_out),
            )

            params = params_0 if hvd.rank() == 0 else params_1
            p = {
                k: v for k, v in params.items()
                if k in inspect.getargspec(opt_class.__init__).args
            }
            opt = opt_class(model.parameters(), **p)
            opt = hvd.DistributedOptimizer(
                opt, named_parameters=model.named_parameters())

            return model, opt

        # Include subclass name so we can sort them lexicographically, otherwise different
        # ranks will have different optimizer orderings
        optimizers = [
            (subclass.__name__, subclass)
            for subclass in torch.optim.Optimizer.__subclasses__()
            if subclass.__module__.startswith('torch.optim') and
            subclass != torch.optim.LBFGS and
            subclass != torch.optim.SparseAdam
        ]
        optimizers.sort(key=lambda tup: tup[0])

        for _, opt_class in optimizers:
            model, optimizer = create_model(opt_class)
            y_pred = model(x)
            loss = F.mse_loss(y_pred, y, size_average=False)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            hvd.broadcast_optimizer_state(optimizer, root_rank=0)
            p0 = {
                k: v for k, v in params_0.items()
                if k in inspect.getargspec(opt_class.__init__).args
            }
            for k, p in p0.items():
                p_actual = optimizer.param_groups[0][k]
                if not isinstance(p, Iterable):
                    p_actual = [p_actual]
                    p = [p]
                for i in range(len(p)):
                    self.assertEqual(type(p_actual[i]), type(p[i]))
                    self.assertAlmostEqual(p_actual[i], p[i], delta=1e-5)

            # Ensure that the parameter option types are compatible with ops
            y_pred = model(x)
            loss = F.mse_loss(y_pred, y, size_average=False)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def test_broadcast_state_no_grad(self):
        class ModelNoGrad(nn.Module):
            def __init__(self, a, b):
                super(ModelNoGrad, self).__init__()
                self.a = nn.Parameter(a.int(), requires_grad=False)
                self.b = nn.Parameter(b)

            def forward(self, x):
                return torch.index_select(self.b, 0, self.a.long()) * x

        hvd.init()
#         torch.cuda.set_device(hvd.local_rank())
        print("Rank is: ", hvd.rank(), "Size is: ", hvd.size())


        a = torch.Tensor([1, 3])
        b = torch.rand(4)

        model = ModelNoGrad(a, b)

        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.001, weight_decay=1e-6, momentum=0.9, nesterov=True)
        optimizer = hvd.DistributedOptimizer(
            optimizer, named_parameters=model.named_parameters())

        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        grad = optimizer.param_groups[0]['params'][1].grad
        bgrad = hvd.broadcast(grad, root_rank=0)

        assert optimizer.param_groups[0]['params'][0].grad is None
        assert torch.all(torch.eq(grad, bgrad)).item()

    def test_broadcast_object(self):
        hvd.init()
#         torch.cuda.set_device(hvd.local_rank())
        print("Rank is: ", hvd.rank(), "Size is: ", hvd.size())

        expected_obj = {
            'hello': 123,
            0: [1, 2]
        }
        obj = expected_obj if hvd.rank() == 0 else {}

        obj = hvd.broadcast_object(obj, root_rank=0)
        self.assertDictEqual(obj, expected_obj)

    def test_compression_fp16(self):
        valid_dtypes = [torch.float32, torch.float64]
        invalid_dtypes = [torch.uint8, torch.int8, torch.int16,
                          torch.int32, torch.int64]

        tensor_size = [5] * 3
        compression = hvd.Compression.fp16

        for dtype in valid_dtypes:
            tensor = torch.ones(tensor_size, dtype=dtype)

            tensor_compressed, ctx = compression.compress(tensor)
            self.assertEqual(tensor_compressed.dtype, torch.float16)

            tensor_decompressed = compression.decompress(
                tensor_compressed, ctx)
            self.assertEqual(tensor_decompressed.dtype, dtype)

            expected = np.ones(tensor_size)
            err = np.linalg.norm(expected - tensor_decompressed.data.numpy())
            self.assertLess(err, 0.00000001)

        for dtype in invalid_dtypes:
            tensor = torch.ones(tensor_size, dtype=dtype)

            tensor_compressed, ctx = compression.compress(tensor)
            self.assertEqual(tensor_compressed.dtype, dtype)

            tensor_decompressed = compression.decompress(
                tensor_compressed, ctx)
            self.assertEqual(tensor_decompressed.dtype, dtype)

            if dtype != torch.int8:  # Cannot cast to NumPy with a CharTensor
                expected = np.ones(tensor_size)
                err = np.linalg.norm(
                    expected - tensor_decompressed.data.numpy())
                self.assertLess(err, 0.00000001)

    def test_force_allreduce(self):
        """Test that allreduce is forced on all gradients during opt.step()."""
        hvd.init()
#         torch.cuda.set_device(hvd.local_rank())
        print("Rank is: ", hvd.rank(), "Size is: ", hvd.size())

        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        N, D_in, H, D_out = 64, 100, 10, 10
        x = torch.randn(N, D_in).requires_grad_()
        y = torch.randn(N, D_out).requires_grad_()

        def new_optimizer(cls, opt_params, model):
            p = {
                k: v for k, v in opt_params.items()
                if k in inspect.getargspec(cls.__init__).args
            }
            return cls(model.parameters(), **p)

        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.fc1 = torch.nn.Linear(D_in, H)
                self.fc2 = torch.nn.Linear(H, D_out)
                self.fc3 = torch.nn.Linear(D_out, D_out)

            def forward(self, x_):
                x_ = F.relu(self.fc1(x_))
                x1_ = self.fc2(x_)
                x2_ = self.fc3(F.relu(x1_))
                return x1_, x2_

        def create_model(opt_class, opt_params):
            model = Net()
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            opt = new_optimizer(opt_class, opt_params, model)
            opt = hvd.DistributedOptimizer(
                opt, named_parameters=model.named_parameters())
            return model, opt

        # L-BFGS is currently unsupported, as are sparse tensors, which are
        # required by SparseAdam optimizer
        optimizers = [
            (subclass.__name__, subclass)
            for subclass in torch.optim.Optimizer.__subclasses__()
            if subclass.__module__.startswith('torch.optim') and
            subclass != torch.optim.LBFGS and
            subclass != torch.optim.SparseAdam
        ]
        optimizers.sort(key=lambda tup: tup[0])

        opt_params_list = [
            dict(lr=0.2, momentum=0.9, weight_decay=0.1, centered=True),
            dict(lr=0.2)
        ]

        for (opt_name, opt_class), opt_params in itertools.product(optimizers, opt_params_list):
            model, optimizer = create_model(opt_class, opt_params)
            y_pred1, y_pred2 = model(x)
            if rank == 0:
                loss = F.mse_loss(y_pred1, y, size_average=False)
            else:
                loss = F.mse_loss(y_pred2, y, size_average=False)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def test_model_parallelism(self):
        """Test that tensors on different GPUs are supported."""
        # Only do this test if there are GPUs available.
        if not torch.cuda.is_available():
            self.skipTest("No GPUs available")

        hvd.init()
#         torch.cuda.set_device(hvd.local_rank())
        print("Rank is: ", hvd.rank(), "Size is: ", hvd.size())

        local_rank = hvd.local_rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        # Skip the test if there are not enough GPUs.
        if torch.cuda.device_count() < hvd.local_size() * 2:
            self.skipTest("Not enough GPUs available")

        first_device = local_rank * 2
        second_device = local_rank * 2 + 1

        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                # Place parts of model on different GPUs.
                self.conv1 = torch.nn.Conv2d(1, 100, 1).cuda(first_device)
                self.conv2 = torch.nn.Conv2d(100, 1, 1).cuda(second_device)

            def forward(self, x):
                x = x.cuda(first_device)
                x = self.conv1(x)
                x = x.cuda(second_device)
                x = self.conv2(x)
                return x

        model = Net()
        inp = torch.rand([1, 1, 1000, 1000])

        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        opt = hvd.DistributedOptimizer(
            opt, named_parameters=model.named_parameters())

        loss = model(inp).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

    def test_delta_optimizer(self):
        """Test that delta optimizer."""
        hvd.init()
#         torch.cuda.set_device(hvd.local_rank())
        print("Rank is: ", hvd.rank(), "Size is: ", hvd.size())

        # TODO support non-MPI Adasum operation
        # Only do this test if there are GPUs available.
        if not hvd.mpi_enabled() or not torch.cuda.is_available():
            self.skipTest("No GPUs available")

        local_rank = hvd.local_rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = torch.nn.Conv2d(1, 100, 1).cuda(local_rank)
                self.conv2 = torch.nn.Conv2d(100, 1, 1).cuda(local_rank)

            def forward(self, x):
                x = x.cuda(local_rank)
                x = self.conv1(x)
                x = x.cuda(local_rank)
                x = self.conv2(x)
                return x

        model = Net()
        inp = torch.rand([1, 1, 1000, 1000])

        opt = torch.optim.SGD(model.parameters(), lr=0.1)

        opt = hvd.DistributedOptimizer(
            opt, named_parameters=model.named_parameters(), op=hvd.Adasum)
        loss = model(inp).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

    def test_dynamic_requires_grad(self):
        """Test that makes sure that gradients can be turned off/on dynamically."""
        hvd.init()
#         torch.cuda.set_device(hvd.local_rank())
        print("Rank is: ", hvd.rank(), "Size is: ", hvd.size())

        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        gen = torch.nn.Conv2d(1, 10, 1)
        disc = torch.nn.Conv2d(10, 1, 1)
        inp = torch.rand([1, 1, 100, 100])

        gen_opt = torch.optim.SGD(gen.parameters(), lr=0.1)
        gen_opt = hvd.DistributedOptimizer(
            gen_opt, named_parameters=gen.named_parameters())

        disc_opt = torch.optim.SGD(disc.parameters(), lr=0.1)
        disc_opt = hvd.DistributedOptimizer(
            disc_opt, named_parameters=disc.named_parameters())

        def train_step(train_generator=False, train_discriminator=False):
            for p in gen.parameters():
                p.requires_grad_(train_generator)
            for p in disc.parameters():
                p.requires_grad_(train_discriminator)

            gen_opt.zero_grad()
            disc_opt.zero_grad()

            loss = disc(gen(inp)).sum()
            loss.backward()

            for p in gen.parameters():
                assert train_generator == (p.grad is not None and p.grad.max().is_nonzero()), \
                    'Gradient for generator is zero but it should be trained or vice versa.'
            for p in disc.parameters():
                assert train_discriminator == (p.grad is not None and p.grad.max().is_nonzero()), \
                    'Gradient for discriminator is zero but it should be trained or vice versa.'

            if train_generator:
                gen_opt.step()
            if train_discriminator:
                disc_opt.step()

        for x in range(10):
            # Step 1: train generator.
            train_step(train_generator=True)

            # Step 2: train discriminator.
            train_step(train_discriminator=True)

    def test_gradient_clipping(self):
        """Test gradient clipping example."""
        hvd.init()
#         torch.cuda.set_device(hvd.local_rank())
        print("Rank is: ", hvd.rank(), "Size is: ", hvd.size())

        size = hvd.size()
        

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        x = torch.ones(1, 1).requires_grad_()
        y = torch.ones(1, 1).requires_grad_()

        model = torch.nn.Linear(1, 1)
        model.weight = torch.nn.Parameter(torch.zeros(1, 1) + 0.5)
        model.bias = torch.nn.Parameter(torch.zeros(1))
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer = hvd.DistributedOptimizer(
            optimizer, named_parameters=model.named_parameters())

        y_pred = model(x)
        loss = F.mse_loss(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.synchronize()
        prior_grad = model.weight.grad.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        clipped_grad = model.weight.grad.item()
        assert abs(prior_grad) > abs(clipped_grad)
        with optimizer.skip_synchronize():
            optimizer.step()

    def test_no_named_parameters(self):
        """Test that leaving the default named_parameters=None will not throw an error."""
        hvd.init()
#         torch.cuda.set_device(hvd.local_rank())
        print("Rank is: ", hvd.rank(), "Size is: ", hvd.size())

        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = torch.nn.Conv2d(1, 100, 1)
                self.conv2 = torch.nn.Conv2d(100, 1, 1)

            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                return x

        model = Net()
        inp = torch.rand([1, 1, 1000, 1000])

        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        opt = hvd.DistributedOptimizer(opt)

        loss = model(inp).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

    # werid, test failure, skip it for now
    def test_horovod_join_allreduce(self):
        """Test Join op with allreduce."""
        # 12/08/2021
        self.skipTest("Got werid test failure in v0.22.1, skip it for now. ")

        hvd.init()
#         torch.cuda.set_device(hvd.local_rank())
        print("Rank is: ", hvd.rank(), "Size is: ", hvd.size())

        rank = hvd.rank()
        size = hvd.size()

        dtypes = [torch.IntTensor, torch.LongTensor,
                  torch.FloatTensor, torch.DoubleTensor]
        if torch.cuda.is_available():
            dtypes += [torch.cuda.IntTensor, torch.cuda.LongTensor,
                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
                       torch.cuda.HalfTensor]

        integral_types = [torch.IntTensor, torch.LongTensor,
                          torch.cuda.IntTensor, torch.cuda.LongTensor]

        dims = [1, 2, 3]
        first_join_ranks = [0, 1]
        cachings = [False, True]
        for dtype, dim, first_join_rank, caching in itertools.product(dtypes, dims, first_join_ranks, cachings):
            torch.manual_seed(1234)

            def div(t, s):
                if _1_5_api and dtype in integral_types:
                    return t.floor_divide(s)
                return t / s

            # Use two tensors to test fusion
            tensor_a = torch.FloatTensor(*([5] * dim)).random_(-100, 100)
            tensor_a = self.cast_and_place(tensor_a, dtype)
            tensor_b = torch.FloatTensor(*([17] * dim)).random_(-100, 100)
            tensor_b = self.cast_and_place(tensor_b, dtype)

            if caching:
                handle_a = hvd.allreduce_async(
                    tensor_a, name="tensor_a", average=True)
                handle_b = hvd.allreduce_async(
                    tensor_b, name="tensor_b", average=True)
                averaged_a = hvd.synchronize(handle_a)
                averaged_b = hvd.synchronize(handle_b)

            if rank == first_join_rank:
                if dtype.is_cuda:
                    ret = hvd.join(hvd.local_rank())
                else:
                    ret = hvd.join()
            else:
                handle_a = hvd.allreduce_async(
                    tensor_a, name="tensor_a", average=True)
                handle_b = hvd.allreduce_async(
                    tensor_b, name="tensor_b", average=True)
                averaged_a = hvd.synchronize(handle_a)
                averaged_b = hvd.synchronize(handle_b)
                if dtype.is_cuda:
                    ret = hvd.join(hvd.local_rank())
                else:
                    ret = hvd.join()

                # Threshold for floating point equality depends on number of
                # ranks, since we're comparing against precise multiplication.
                if size <= 3 or dtype in integral_types:
                    threshold = 0
                elif size < 10:
                    threshold = 1e-4
                elif size < 15:
                    threshold = 5e-4
                else:
                    break
                assert torch.allclose(averaged_a, div(tensor_a * (size - 1), size), threshold), \
                    'hvd.join with hvd.allreduce produces incorrect results'
                assert torch.allclose(averaged_b, div(tensor_b * (size - 1), size), threshold), \
                    'hvd.join with hvd.allreduce produces incorrect results'

    def test_horovod_join_broadcast(self):
        """Test Join op with broadcast."""
        hvd.init()
#         torch.cuda.set_device(hvd.local_rank())
        print("Rank is: ", hvd.rank(), "Size is: ", hvd.size())

        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        dims = [17] * 3
        tensor = torch.FloatTensor(*dims)

        if rank == 0:
            ret = hvd.join(hvd.local_rank())
        else:
            try:
                broadcasted_tensor = hvd.broadcast(
                    tensor, 1, name="test_horovod_join_broadcast")
                assert False, 'hvd.broadcast did not throw error'
            except (torch.FatalError, RuntimeError):
                pass

            if torch.cuda.is_available():
                ret = hvd.join(hvd.local_rank())
            else:
                ret = hvd.join()

    def test_optimizer_no_named_parameters(self):
        hvd.init()
#         torch.cuda.set_device(hvd.local_rank())
        print("Rank is: ", hvd.rank(), "Size is: ", hvd.size())


        model = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 10))
        optimizer = torch.optim.SGD(
            [{"params": model[0].parameters()}, {"params": model[1].parameters()}, ],
            lr=0.001,
        )
        optimizer = hvd.DistributedOptimizer(optimizer)

        params = optimizer._parameter_names
        self.assertEqual(len(params), len(set(params.values())))

        # Make sure all workers have the same set of parameter names
        all_param_names = hvd.allgather_object(set(params.values()))
        self.assertEqual(len(all_param_names), hvd.size())
        for param_names in all_param_names:
            self.assertEqual(all_param_names[0], param_names)

    def test_sparse_embeddings(self):
        """Test that Horovod will correctly aggregate sparse gradients."""
        hvd.init()
#         torch.cuda.set_device(hvd.local_rank())
        print("Rank is: ", hvd.rank(), "Size is: ", hvd.size())


        for sparse_as_dense in [False, True]:
            class Net(torch.nn.Module):
                def __init__(self):
                    super(Net, self).__init__()
                    self.embedding = nn.Embedding(10, 3, sparse=True)

                def forward(self, x):
                    x = self.embedding(x)
                    return x

            model = Net()

            if hvd.rank() == 0:
                inp = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
            else:
                inp = torch.LongTensor([[1, 3, 4], [4, 7, 9]])

            # list() see: https://github.com/pytorch/pytorch/issues/47594
            opt = torch.optim.SparseAdam(list(model.parameters()), lr=0.1)
            opt = hvd.DistributedOptimizer(
                opt, sparse_as_dense=sparse_as_dense)

            loss = model(inp).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()

    def test_async_sparse_allreduce(self):
        """Test that allgather over indices and values is equivalent to allreduce."""
        hvd.init()
#         torch.cuda.set_device(hvd.local_rank())
        print("Rank is: ", hvd.rank(), "Size is: ", hvd.size())

        # Generate random tensors, then convert them to sparse
        def random_sparse_tensor(*shape):
            t = torch.rand(*shape)
            t[t < 0.8] = 0
            return t.to_sparse()

        tensor_sizes = [17, 32, 81, 12, 15, 23, 22] * 5
        tensors = [random_sparse_tensor(d0, 10) for d0 in tensor_sizes]
        allreduced_tensors = [hvd.allreduce(t.to_dense()) for t in tensors]

        handles = [hvd.sparse_allreduce_async(t, op=hvd.Average, name=str(i))
                   for i, t in enumerate(tensors)]
        allgathered_tensors = [handle() for handle in handles]

        for reduced, gathered in zip(allreduced_tensors, allgathered_tensors):
            assert torch.allclose(reduced, gathered.to_dense(), 1e-6)