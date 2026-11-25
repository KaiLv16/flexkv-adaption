#pragma once

#include <cstring>

#define CUDART_VERSION 12000

enum {
  cudaSuccess=0, cudaDevAttrGPUDirectRDMAWritesOrdering=118,
  cudaDevAttrGPUDirectRDMAFlushWritesOptions=117,
  cudaFlushGPUDirectRDMAWritesToAllDevices=200, cudaFlushGPUDirectRDMAWritesOptionHost=1,
  cudaFlushGPUDirectRDMAWritesTargetCurrentDevice=0
};
struct cudaPointerAttributes { void *hostPointer; };
namespace fake {
static void *mapped_address = nullptr;
}
inline int cudaSetDevice(int) { return cudaSuccess; }
inline int cudaGetLastError() { return cudaSuccess; }
inline int cudaDeviceGetPCIBusId(char *bus, int size, int) {
  std::strncpy(bus, "0000:00:00.0", size);
  return cudaSuccess;
}
inline int cudaPointerGetAttributes(cudaPointerAttributes *attr, const void *) {
  attr->hostPointer = fake::mapped_address;
  return cudaSuccess;
}
inline int cudaDeviceGetAttribute(int *value, int attr, int) {
  *value = attr == cudaDevAttrGPUDirectRDMAWritesOrdering ? 0 : 1;
  return cudaSuccess;
}
inline int cudaDeviceFlushGPUDirectRDMAWrites(int, int) { return cudaSuccess; }
inline int cudaDeviceSynchronize() { return cudaSuccess; }
