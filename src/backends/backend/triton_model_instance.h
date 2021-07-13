// Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#pragma once

#include <functional>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include "model_config.pb.h"
#include "src/core/constants.h"
#include "src/core/memory.h"
#include "src/core/metric_model_reporter.h"
#include "src/core/server_message.h"
#include "src/core/status.h"
#include "triton/common/sync_queue.h"

namespace nvidia { namespace inferenceserver {

class TritonModel;
class InferenceRequest;

//
// Represents a model instance.
//
class TritonModelInstance {
 public:
  static Status CreateInstances(
      TritonModel* model, const HostPolicyCmdlineConfigMap& host_policy_map,
      const inference::ModelConfig& model_config, const bool device_blocking);
  ~TritonModelInstance();

  const std::string& Name() const { return name_; }
  size_t Index() const { return index_; }
  TRITONSERVER_InstanceGroupKind Kind() const { return kind_; }
  int32_t DeviceId() const { return device_id_; }
  const HostPolicyCmdlineConfig& HostPolicy() const { return host_policy_; }
  const TritonServerMessage& HostPolicyMessage() const
  {
    return host_policy_message_;
  }
  bool IsPassive() const { return passive_; }
  const std::vector<std::string>& Profiles() const { return profile_names_; }

  Status Initialize();
  Status WarmUp();
  void Schedule(
      std::vector<std::unique_ptr<InferenceRequest>>&& requests,
      const std::function<void()>& OnCompletion);

  TritonModel* Model() const { return model_; }
  void* State() { return state_; }
  void SetState(void* state) { state_ = state; }

  MetricModelReporter* MetricReporter() const { return reporter_.get(); }

 private:
  DISALLOW_COPY_AND_ASSIGN(TritonModelInstance);
  class TritonBackendThread;
  TritonModelInstance(
      TritonModel* model, const std::string& name, const size_t index,
      const TRITONSERVER_InstanceGroupKind kind, const int32_t device_id,
      const std::vector<std::string>& profile_names, const bool passive,
      const HostPolicyCmdlineConfig& host_policy,
      const TritonServerMessage& host_policy_message);
  static Status CreateInstance(
      TritonModel* model, const std::string& name, const size_t index,
      const TRITONSERVER_InstanceGroupKind kind, const int32_t device_id,
      const std::vector<std::string>& profile_names, const bool passive,
      const std::string& host_policy_name,
      const HostPolicyCmdlineConfig& host_policy,
      const inference::ModelRateLimiter& rate_limiter_config,
      const bool device_blocking,
      std::map<uint32_t, std::shared_ptr<TritonBackendThread>>*
          device_to_thread_map);
  Status SetBackendThread(
      const TRITONSERVER_InstanceGroupKind kind, const int32_t device_id,
      const bool device_blocking,
      std::map<uint32_t, std::shared_ptr<TritonBackendThread>>*
          device_to_thread_map);
  Status GenerateWarmupData();

  void Execute(std::vector<TRITONBACKEND_Request*>& triton_requests);

  class TritonBackendThread {
   public:
    static Status CreateBackendThread(
        const std::string name, TritonModelInstance* model, const int nice,
        const int32_t device_id,
        std::unique_ptr<TritonBackendThread>* triton_backend_thread);
    void AddModelInstance(TritonModelInstance* model_instance);
    Status InitAndWarmUpModelInstance(TritonModelInstance* model_instance);
    ~TritonBackendThread();

   private:
    TritonBackendThread(const std::string& name, TritonModel* model);
    void BackendThread(const int nice, const int32_t device_id);

    std::string name_;

    TritonModel* model_;
    std::deque<TritonModelInstance*> model_instances_;

    std::thread backend_thread_;
    std::atomic<bool> backend_thread_exit_;
  };
  std::shared_ptr<TritonBackendThread> triton_backend_thread_;

  struct WarmupData {
    WarmupData(const std::string& sample_name) : sample_name_(sample_name) {}

    std::string sample_name_;
    std::vector<std::unique_ptr<InferenceRequest>> requests_;

    // Placeholder for input data
    std::unique_ptr<AllocatedMemory> zero_data_;
    std::unique_ptr<AllocatedMemory> random_data_;
    std::vector<std::unique_ptr<std::string>> provided_data_;
  };
  std::vector<WarmupData> warmup_samples_;

  // The TritonModel object that owns this instance. The instance
  // holds this as a raw pointer because the lifetime of the model is
  // guaranteed to be longer than the lifetime of an instance owned by the
  // model.
  TritonModel* model_;

  std::string name_;
  size_t index_;

  // For CPU device_id_ is always 0. For GPU device_id_ indicates the
  // GPU device to be used by the instance.
  TRITONSERVER_InstanceGroupKind kind_;
  int32_t device_id_;
  const HostPolicyCmdlineConfig host_policy_;
  TritonServerMessage host_policy_message_;
  std::vector<std::string> profile_names_;
  bool passive_;

  std::shared_ptr<TritonBackendThread> backend_thread_;

  // Reporter for metrics, or nullptr if no metrics should be reported
  std::shared_ptr<MetricModelReporter> reporter_;

  // Opaque state associated with this model instance.
  void* state_;
};

}}  // namespace nvidia::inferenceserver
