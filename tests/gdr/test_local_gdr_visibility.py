"""Opt-in hardware check for GDR overwrites of GPU blocks recently read by a kernel."""

import os
import select
import unittest
from concurrent.futures import ThreadPoolExecutor


@unittest.skipUnless(os.environ.get("FLEXKV_RUN_GDR_VISIBILITY_TEST") == "1",
                     "requires a configured local GPU and RDMA NIC")
class LocalGdrVisibilityTest(unittest.TestCase):
    def test_reused_blocks_are_visible_at_layer_notification(self):
        import torch
        from flexkv import c_ext
        from flexkv.transfer import local_gdr

        self.assertTrue(local_gdr.enabled("h2d"), "set FLEXKV_LOCAL_GDR=h2d")
        device = torch.cuda.current_device()
        blocks, width = 8, 512
        host_ids = torch.tensor([6, 1, 7, 3], dtype=torch.int64)
        gpu_ids = torch.tensor([0, 5, 2, 7], dtype=torch.int64)
        pattern = torch.arange(blocks * width, dtype=torch.int64).reshape(blocks, width)
        host = torch.empty_like(pattern)
        expected = pattern.clone()
        gpu = expected.to(torch.device("cuda", device))

        region = c_ext.RegionDesc()
        region.name = "visibility"
        region.cpu_ptr = host.data_ptr()
        region.cpu_block_stride_in_bytes = width * host.element_size()
        region.cpu_num_blocks = blocks
        region.gpu_block_ptrs_flat = [gpu.data_ptr()]
        region.gpu_block_strides_in_bytes = [region.cpu_block_stride_in_bytes]
        region.gpu_chunk_sizes_in_bytes = [region.cpu_block_stride_in_bytes]
        region.gpu_layer_strides_in_bytes = [0]
        region.gpu_kv_strides_in_bytes = [0]
        region.num_tensors_per_gpu = region.num_layers = 1
        region.gpu_num_blocks = [blocks]
        batch = c_ext.RegionBatchGroup([device], [region])
        owner = local_gdr.Owner([gpu], 0)
        gpu._flexkv_gdr_view = owner.metadata(0)
        self.addCleanup(owner.retire)
        fd = os.eventfd(0, os.EFD_CLOEXEC | os.EFD_NONBLOCK)
        self.addCleanup(os.close, fd)
        batch.set_layer_eventfds(torch.tensor([fd], dtype=torch.int32), 1, 1)
        request = c_ext.RegionRequest()
        request.cpu_block_id_tensor = host_ids
        request.gpu_block_id_tensor = gpu_ids
        request.is_host_to_device = True
        request.layer_granularity = 1
        request.milestone_layer = 0

        leases = local_gdr.attach_routes(batch, [device], [gpu], [host])
        unsafe = False
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                for generation in range(1, 33):
                    # Warm the same lines via a kernel, then read them via a
                    # kernel again as soon as the layer event arrives.
                    self.assertTrue(torch.equal((gpu ^ 123).cpu(), expected ^ 123))
                    host.copy_(pattern + generation * blocks * width)
                    expected[gpu_ids] = host[host_ids]
                    transfer = pool.submit(batch.submit_layerwise, [request], [], 0)
                    if not select.select([fd], [], [], 30)[0]:
                        batch.stop_gdr()
                        transfer.result()
                        self.fail("missing layer event")
                    self.assertEqual(os.eventfd_read(fd), 1)
                    observed = (gpu ^ 123).cpu()
                    transfer.result()
                    self.assertTrue(torch.equal(observed, expected ^ 123), f"generation {generation}")
            self.assertGreater(batch.gdr_counters()[0][0], 0)
        except c_ext.UnsafeGdrTransfer:
            unsafe = True
            local_gdr.quarantine(leases)
            raise
        finally:
            if not unsafe:
                local_gdr.close_routes(batch, leases)


if __name__ == "__main__":
    unittest.main()
