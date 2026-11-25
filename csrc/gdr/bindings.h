// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "gdr/transport.h"
#include "region_batch.h"

#include <c10/cuda/CUDAGuard.h>
#include <pybind11/stl.h>

namespace flexkv::gdr {
namespace py = pybind11;

struct PendingSender {
  std::unique_ptr<Transport> transport;
};

inline void bind(py::module_ &m, py::class_<RegionBatchGroup> &batch) {
  py::register_exception<UnsafeTransfer>(m, "UnsafeGdrTransfer");
  py::class_<Transport>(m, "GdrOwnerEndpoint")
      .def(py::init([](int device, const std::string &nic, int depth) {
        const c10::cuda::CUDAGuard guard(device);
        return std::make_unique<Transport>(device, false, nic, "", depth, 0);
      }))
      .def("export_tensor", [](Transport &transport, const torch::Tensor &tensor) {
        TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous(),
                    "GDR requires a contiguous device view");
        const c10::cuda::CUDAGuard guard(tensor.device());
        transport.register_device(tensor.storage().data_ptr().get(), tensor.storage().nbytes());
        const auto region = transport.export_region(tensor.data_ptr(), tensor.nbytes());
        return py::make_tuple(region.remote, region.bytes, region.key);
      })
      .def_property_readonly("nic", &Transport::nic)
      .def_property_readonly("endpoint", &Transport::endpoint)
      .def_property_readonly("read_depth", &Transport::read_depth)
      .def("accept", &Transport::accept)
      .def("release", &Transport::release);
  py::class_<PendingSender>(m, "GdrSender")
      .def(py::init([](int device, const std::string &nic,
                       const std::string &endpoint, int depth, int readbacks) {
        const c10::cuda::CUDAGuard guard(device);
        return PendingSender{
            std::make_unique<Transport>(device, true, nic, endpoint, depth, readbacks)};
      }))
      .def_property_readonly("qpn", [](PendingSender &sender) {
        return sender.transport->sender_qpn();
      })
      .def("register_host", [](PendingSender &sender, const torch::Tensor &tensor) {
        TORCH_CHECK(tensor.device().is_cpu() && tensor.is_contiguous(),
                    "GDR requires a contiguous Host pool");
        sender.transport->register_host(tensor.data_ptr(), tensor.nbytes());
      })
      .def("connect", [](PendingSender &sender, uint32_t target,
                         const std::vector<std::tuple<uintptr_t, size_t, uint64_t, uint32_t>> &views) {
        std::vector<RemoteRegion> regions;
        regions.reserve(views.size());
        for (const auto &[local, bytes, remote, key] : views)
          regions.push_back({local, bytes, remote, key});
        sender.transport->connect(target, std::move(regions));
      });

  m.def("gdr_producer_ready", [](int device) {
    const c10::cuda::CUDAGuard guard(device);
    TORCH_CHECK(cudaDeviceSynchronize() == cudaSuccess, "GDR producer sync failed");
  }, py::call_guard<py::gil_scoped_release>());
  m.def("gdr_device_visible", [](int device) {
    const c10::cuda::CUDAGuard guard(device);
    Transport::make_device_visible(device);
  }, py::call_guard<py::gil_scoped_release>());
  batch.def("install_gdr_route", [](RegionBatchGroup &group, int rank, PendingSender &sender, int fd) {
        group.install_gdr_route(rank, std::move(sender.transport), fd);
      })
      .def("close_gdr", &RegionBatchGroup::close_gdr)
      .def("stop_gdr", &RegionBatchGroup::stop_gdr)
      .def("gdr_counters", &RegionBatchGroup::gdr_counters);
}
}  // namespace flexkv::gdr
