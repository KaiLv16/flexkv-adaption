// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "gdr/transport.h"
#include "region_batch.h"

#include <c10/cuda/CUDAGuard.h>
#include <cerrno>
#include <chrono>
#include <pybind11/stl.h>
#include <sys/socket.h>
#include <tuple>

namespace flexkv {
namespace gdr {
namespace py = pybind11;

struct PendingSender {
  std::unique_ptr<Transport> transport;
};

inline void bind(py::module_ &m, py::class_<RegionBatchGroup> &batch) {
#ifdef FLEXKV_GDR_MAPPED
  m.attr("gdr_uses_mapped_memory") = true;
#else
  m.attr("gdr_uses_mapped_memory") = false;
#endif
  py::register_exception<UnsafeTransfer>(m, "UnsafeGdrTransfer");
  py::class_<Transport>(m, "GdrOwnerEndpoint")
      .def(py::init([](int device, const std::string &nic, int depth, size_t qp_count) {
        const c10::cuda::CUDAGuard guard(device);
        return std::unique_ptr<Transport>(new Transport(device, false, nic, "", depth, 0, qp_count));
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
                       const std::string &endpoint, int depth, int readbacks, size_t qp_count,
                       size_t h2d_qp_count, size_t d2h_qp_count, int max_sge) {
        const c10::cuda::CUDAGuard guard(device);
        return PendingSender{
            std::unique_ptr<Transport>(new Transport(
                device, true, nic, endpoint, depth, readbacks,
                qp_count, h2d_qp_count, d2h_qp_count, max_sge))};
      }), py::arg("device"), py::arg("nic"), py::arg("endpoint"),
          py::arg("depth"), py::arg("readbacks"), py::arg("qp_count"),
          py::arg("h2d_qp_count") = 0, py::arg("d2h_qp_count") = 0, py::arg("max_sge") = 0)
      .def_property_readonly("qpns", [](PendingSender &sender) {
        return sender.transport->sender_qpns();
      })
      .def_property_readonly("h2d_qp_count", [](PendingSender &sender) {
        return sender.transport->h2d_qp_count();
      })
      .def_property_readonly("d2h_qp_count", [](PendingSender &sender) {
        return sender.transport->d2h_qp_count();
      })
      .def_property_readonly("write_sge_limit", [](PendingSender &sender) {
        return sender.transport->write_sge_limit();
      })
      .def_property_readonly("read_sge_limit", [](PendingSender &sender) {
        return sender.transport->read_sge_limit();
      })
      .def_property_readonly("uses_wr_api", [](PendingSender &sender) {
        return sender.transport->uses_wr_api();
      })
      .def("register_host", [](PendingSender &sender, const torch::Tensor &tensor) {
        TORCH_CHECK(tensor.device().is_cpu() && tensor.is_contiguous(),
                    "GDR requires a contiguous Host pool");
        sender.transport->register_host(tensor.data_ptr(), tensor.nbytes());
      })
      .def("connect", [](PendingSender &sender, const std::vector<uint32_t> &targets,
                         const std::vector<std::tuple<uintptr_t, size_t, uint64_t, uint32_t>> &views) {
        std::vector<RemoteRegion> regions;
        regions.reserve(views.size());
        for (const auto &view : views)
          regions.push_back({std::get<0>(view), std::get<1>(view),
                             std::get<2>(view), std::get<3>(view)});
        sender.transport->connect(targets, std::move(regions));
      });

  m.def("gdr_producer_ready", [](int device) {
    const c10::cuda::CUDAGuard guard(device);
    TORCH_CHECK(cudaDeviceSynchronize() == cudaSuccess, "GDR producer sync failed");
  }, py::call_guard<py::gil_scoped_release>());
  m.def("gdr_device_visible", [](int device) {
    const c10::cuda::CUDAGuard guard(device);
    Transport::make_device_visible(device);
  }, py::call_guard<py::gil_scoped_release>());
  m.def("gdr_serve_owner_commands", [](int fd, int device, bool profile) {
    const c10::cuda::CUDAGuard guard(device);
    using Clock = std::chrono::steady_clock;
    auto elapsed_ns = [](Clock::time_point from, Clock::time_point to) -> uint64_t {
      return to > from ? static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(to - from).count()) : 0ULL;
    };
    uint64_t r_calls = 0, r_sync_ns = 0, r_total_ns = 0;
    uint64_t v_calls = 0, v_sync_ns = 0, v_total_ns = 0;
    while (true) {
      char cmd = 0;
      ssize_t received;
      do {
        received = ::recv(fd, &cmd, 1, 0);
      } while (received < 0 && errno == EINTR);
      if (received == 0) {
        return std::make_tuple(false, r_calls, r_sync_ns, r_total_ns, v_calls, v_sync_ns, v_total_ns);
      }
      if (received < 0) {
        throw std::runtime_error("GDR: owner command recv failed");
      }
      if (cmd == 'D') {
        return std::make_tuple(true, r_calls, r_sync_ns, r_total_ns, v_calls, v_sync_ns, v_total_ns);
      }
      Clock::time_point t0{}, t1{};
      if (profile) t0 = Clock::now();
      if (cmd == 'R') {
        TORCH_CHECK(cudaDeviceSynchronize() == cudaSuccess, "GDR producer sync failed");
      } else if (cmd == 'V') {
        Transport::make_device_visible(device);
      } else {
        return std::make_tuple(false, r_calls, r_sync_ns, r_total_ns, v_calls, v_sync_ns, v_total_ns);
      }
      if (profile) t1 = Clock::now();
      const char ack = 'A';
      ssize_t sent;
      do {
        sent = ::send(fd, &ack, 1, MSG_NOSIGNAL);
      } while (sent < 0 && errno == EINTR);
      if (sent != 1) {
        throw std::runtime_error("GDR: owner acknowledgement failed");
      }
      if (profile) {
        const auto t2 = Clock::now();
        if (cmd == 'R') {
          ++r_calls;
          r_sync_ns += elapsed_ns(t0, t1);
          r_total_ns += elapsed_ns(t0, t2);
        } else {
          ++v_calls;
          v_sync_ns += elapsed_ns(t0, t1);
          v_total_ns += elapsed_ns(t0, t2);
        }
      }
    }
  }, py::call_guard<py::gil_scoped_release>());
  batch.def("install_gdr_route", [](RegionBatchGroup &group, int rank, PendingSender &sender, int fd) {
        group.install_gdr_route(rank, std::move(sender.transport), fd);
      })
      .def("close_gdr", &RegionBatchGroup::close_gdr)
      .def("stop_gdr", &RegionBatchGroup::stop_gdr)
      .def("begin_d2h_batch", [](RegionBatchGroup &group) { group.begin_d2h_batch(); })
      .def("end_d2h_batch", [](RegionBatchGroup &group) { group.end_d2h_batch(); })
      .def("gdr_counters", &RegionBatchGroup::gdr_counters)
      .def("gdr_profile", &RegionBatchGroup::gdr_profile);
}
}  // namespace gdr
}  // namespace flexkv
