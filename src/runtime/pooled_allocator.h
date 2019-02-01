/*!
 *  Copyright (c) 2019 by Contributors
 * \file runtime/pooled_allocator.h
 */
#ifndef TVM_RUNTIME_POOLED_ALLOCATOR_H_
#define TVM_RUNTIME_POOLED_ALLOCATOR_H_

#include <atomic>
#include <mutex>
#include <unordered_map>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/memory_manager.h>

namespace tvm {
namespace runtime {

class PooledAllocator final : public Allocator {
 public:
  static constexpr size_t kDefaultPageSize = 4096;
  
  PooledAllocator(TVMContext ctx, size_t page_size=kDefaultPageSize) :
      Allocator(ctx), page_size_(page_size), used_memory_(0) {}

  ~PooledAllocator() {
    ReleaseAll();
  }
  
  Buffer Alloc(size_t nbytes, size_t alignment, TVMType type_hint) override {
    std::lock_guard<std::mutex> lock(mu_);
    size_t size = ((nbytes + page_size_ - 1) / page_size_) * page_size_;
    auto&& it = memory_pool_.find(size);
    if (it != memory_pool_.end() && !it->second.empty()) {
      auto&& pool = it->second;
      auto ret = pool.back();
      pool.pop_back();
      return ret;
    }
    Buffer buf;
    buf.ctx = ctx_;
    buf.size = size;
    buf.data = DeviceAPI::Get(ctx_)->AllocDataSpace(
        ctx_, size, alignment, type_hint);
    used_memory_.fetch_add(size, std::memory_order_relaxed);
    LOG(INFO) << "allocate " << size << " B, used memory " << used_memory_ << " B";
    return buf;
  }

  void Free(const Buffer& buffer) override {
    std::lock_guard<std::mutex> lock(mu_);
    if (memory_pool_.find(buffer.size) == memory_pool_.end()) {
      memory_pool_.emplace(buffer.size, std::vector<Buffer>{});
    }
    memory_pool_.at(buffer.size).push_back(buffer);
    LOG(INFO) << "reclaim buffer " << buffer.size;
  }

  size_t UsedMemory() override {
    return used_memory_.load(std::memory_order_relaxed);
  }

 private:
  void ReleaseAll() {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto const& it : memory_pool_) {
      auto const& pool = it.second;
      for (auto const& buf : pool) {
        DeviceAPI::Get(buf.ctx)->FreeDataSpace(buf.ctx, buf.data);
      }
    }
    memory_pool_.clear();
    used_memory_ = 0;
    LOG(INFO) << "release all buffers";
  }

 private:
  size_t page_size_;
  std::atomic<size_t> used_memory_;
  std::unordered_map<size_t, std::vector<Buffer>> memory_pool_;
  std::mutex mu_;
};

}  // namespace runtime
}  // namespace tvm

#endif // TVM_RUNTIME_POOLED_ALLOCATOR_H_
