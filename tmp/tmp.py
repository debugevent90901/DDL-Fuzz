from horovod.common import mpi_env_rank_and_size, skip_or_fail_gpu_test, temppath
from distutils.version import LooseVersion

import os, sys, time, inspect, platform, warnings, unittest, itertools

from collections.abc import Iterable
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import horovod.torch as hvd

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, 'utils'))

def test_horovod_allreduce(self):
        """Test that the allreduce correctly sums 1D, 2D, 3D tensors."""
        hvd.init()
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

    def test_horovod_grouped_allreduce_grad_average(self):
        """Test the correctness of the grouped allreduce averaged gradient."""
        hvd.init()
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

    
    def test_horovod_join_allreduce(self):
        """Test Join op with allreduce."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        dtypes = [torch.IntTensor, torch.LongTensor,
                  torch.FloatTensor, torch.DoubleTensor]
        if torch.cuda.is_available():
            dtypes += [torch.cuda.IntTensor, torch.cuda.LongTensor,
                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
                       torch.cuda.HalfTensor]

        integral_types = [torch.IntTensor, torch.LongTensor, torch.cuda.IntTensor, torch.cuda.LongTensor]

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
                handle_a = hvd.allreduce_async(tensor_a, name="tensor_a", average=True)
                handle_b = hvd.allreduce_async(tensor_b, name="tensor_b", average=True)
                averaged_a = hvd.synchronize(handle_a)
                averaged_b = hvd.synchronize(handle_b)

            if rank == first_join_rank:
                if dtype.is_cuda:
                    ret = hvd.join(hvd.local_rank())
                else:
                    ret = hvd.join()
            else:
                handle_a = hvd.allreduce_async(tensor_a, name="tensor_a", average=True)
                handle_b = hvd.allreduce_async(tensor_b, name="tensor_b", average=True)
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
                assert torch.allclose(averaged_b, div(tensor_b * (size - 1), 