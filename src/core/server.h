// Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

#include <stddef.h>
#include <stdint.h>
#include <atomic>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "model_config.pb.h"
#include "src/core/model_config.h"
#include "src/core/model_repository_manager.h"
#include "src/core/persistent_backend_manager.h"
#include "src/core/rate_limiter.h"
#include "src/core/status.h"

namespace nvidia { namespace inferenceserver {

class InferenceBackend;
class InferenceRequest;

enum class ModelControlMode { MODE_NONE, MODE_POLL, MODE_EXPLICIT };

enum class RateLimitMode { RL_EXEC_COUNT, RL_OFF };

// Readiness status for the inference server.
enum class ServerReadyState {
  // The server is in an invalid state and will likely not response
  // correctly to any requests.
  SERVER_INVALID,

  // The server is initializing.
  SERVER_INITIALIZING,

  // The server is ready and accepting requests.
  SERVER_READY,

  // The server is exiting and will not respond to requests.
  SERVER_EXITING,

  // The server did not initialize correctly.
  SERVER_FAILED_TO_INITIALIZE
};

// Inference server information.
class InferenceServer {
 public:
  // Construct an inference server.
  InferenceServer();

  // Initialize the server. Return true on success, false otherwise.
  Status Init();

  // Stop the server.  Return true if all models are unloaded, false
  // if exit timeout occurs. If 'force' is true attempt to stop the
  // server even if it is not in a ready state.
  Status Stop(const bool force = false);

  // Check the model repository for changes and update server state
  // based on those changes.
  Status PollModelRepository();

  // Server health
  Status IsLive(bool* live);
  Status IsReady(bool* ready);

  // Model health
  Status ModelIsReady(
      const std::string& model_name, const int64_t model_version, bool* ready);

  // Return the ready versions of specific model
  Status ModelReadyVersions(
      const std::string& model_name, std::vector<int64_t>* versions);

  // Return the ready versions of all models
  Status ModelReadyVersions(
      std::map<std::string, std::vector<int64_t>>* model_versions);

  /// Get the index of all models in all repositories.
  /// \param ready_only If true return only index of models that are ready.
  /// \param index Returns the index.
  /// \return error status.
  Status RepositoryIndex(
      const bool ready_only,
      std::vector<ModelRepositoryManager::ModelIndex>* index);

  // Inference. If Status::Success is returned then this function has
  // taken ownership of the request object and so 'request' will be
  // nullptr. If non-success is returned then the caller still retains
  // ownership of 'request'.
  Status InferAsync(std::unique_ptr<InferenceRequest>& request);

  // Load the corresponding model. Reload the model if it has been loaded.
  Status LoadModel(const std::string& model_name);

  // Unload the corresponding model.
  Status UnloadModel(
      const std::string& model_name, const bool unload_dependents);

  // Print backends and models summary
  Status PrintBackendAndModelSummary();

  // Return the server version.
  const std::string& Version() const { return version_; }

  // Return the server extensions.
  const std::vector<const char*>& Extensions() const { return extensions_; }

  // Get / set the ID of the server.
  const std::string& Id() const { return id_; }
  void SetId(const std::string& id) { id_ = id; }

  // Get / set the model repository path
  const std::set<std::string>& ModelRepositoryPaths() const
  {
    return model_repository_paths_;
  }

  void SetModelRepositoryPaths(const std::set<std::string>& p)
  {
    model_repository_paths_ = p;
  }

  // Get / set model control mode.
  ModelControlMode GetModelControlMode() const { return model_control_mode_; }
  void SetModelControlMode(ModelControlMode m) { model_control_mode_ = m; }

  // Get / set the startup models
  const std::set<std::string>& StartupModels() const { return startup_models_; }
  void SetStartupModels(const std::set<std::string>& m) { startup_models_ = m; }

  // Get / set strict model configuration enable.
  bool StrictModelConfigEnabled() const { return strict_model_config_; }
  void SetStrictModelConfigEnabled(bool e) { strict_model_config_ = e; }

  // Get / set rate limiter mode.
  RateLimitMode RateLimiterMode() const { return rate_limit_mode_; }
  void SetRateLimiterMode(RateLimitMode m) { rate_limit_mode_ = m; }

  // Get / set rate limit resource counts
  const RateLimiter::ResourceMap& RateLimiterResources() const
  {
    return rate_limit_resource_map_;
  }
  void SetRateLimiterResources(const RateLimiter::ResourceMap& rm)
  {
    rate_limit_resource_map_ = rm;
  }

  // Get / set the pinned memory pool byte size.
  int64_t PinnedMemoryPoolByteSize() const { return pinned_memory_pool_size_; }
  void SetPinnedMemoryPoolByteSize(int64_t s)
  {
    pinned_memory_pool_size_ = std::max((int64_t)0, s);
  }

  // Get / set CUDA memory pool size
  const std::map<int, uint64_t>& CudaMemoryPoolByteSize() const
  {
    return cuda_memory_pool_size_;
  }

  void SetCudaMemoryPoolByteSize(const std::map<int, uint64_t>& s)
  {
    cuda_memory_pool_size_ = s;
  }

  // Get / set the minimum support CUDA compute capability.
  double MinSupportedComputeCapability() const
  {
    return min_supported_compute_capability_;
  }
  void SetMinSupportedComputeCapability(double c)
  {
    min_supported_compute_capability_ = c;
  }

  // Get / set strict readiness enable.
  bool StrictReadinessEnabled() const { return strict_readiness_; }
  void SetStrictReadinessEnabled(bool e) { strict_readiness_ = e; }

  // Get / set the server exit timeout, in seconds.
  int32_t ExitTimeoutSeconds() const { return exit_timeout_secs_; }
  void SetExitTimeoutSeconds(int32_t s) { exit_timeout_secs_ = std::max(0, s); }

  void SetBufferManagerThreadCount(unsigned int c)
  {
    buffer_manager_thread_count_ = c;
  }

  // Set a backend command-line configuration
  void SetBackendCmdlineConfig(const BackendCmdlineConfigMap& bc)
  {
    backend_cmdline_config_map_ = bc;
  }

  void SetHostPolicyCmdlineConfig(const HostPolicyCmdlineConfigMap& hp)
  {
    host_policy_map_ = hp;
  }

  void SetRepoAgentDir(const std::string& d) { repoagent_dir_ = d; }

  // FIXME TF specific functions should be removed once all backends
  // use BackendConfig.

  // Get / set Tensorflow soft placement enable.
  bool TensorFlowSoftPlacementEnabled() const
  {
    return tf_soft_placement_enabled_;
  }
  void SetTensorFlowSoftPlacementEnabled(bool e)
  {
    tf_soft_placement_enabled_ = e;
  }

  // Get / set Tensorflow GPU memory fraction.
  float TensorFlowGPUMemoryFraction() const { return tf_gpu_memory_fraction_; }
  void SetTensorFlowGPUMemoryFraction(float f) { tf_gpu_memory_fraction_ = f; }

  // Return the requested InferenceBackend object.
  Status GetInferenceBackend(
      const std::string& model_name, const int64_t model_version,
      std::shared_ptr<InferenceBackend>* backend)
  {
    if (ready_state_ != ServerReadyState::SERVER_READY) {
      return Status(Status::Code::UNAVAILABLE, "Server not ready");
    }
    return model_repository_manager_->GetInferenceBackend(
        model_name, model_version, backend);
  }

  // Return the pointer to RateLimiter object.
  std::shared_ptr<RateLimiter> GetRateLimiter() { return rate_limiter_; }

 private:
  const std::string version_;
  std::string id_;
  std::vector<const char*> extensions_;

  std::set<std::string> model_repository_paths_;
  std::set<std::string> startup_models_;
  ModelControlMode model_control_mode_;
  bool strict_model_config_;
  bool strict_readiness_;
  uint32_t exit_timeout_secs_;
  uint32_t buffer_manager_thread_count_;
  uint64_t pinned_memory_pool_size_;
  std::map<int, uint64_t> cuda_memory_pool_size_;
  double min_supported_compute_capability_;
  BackendCmdlineConfigMap backend_cmdline_config_map_;
  HostPolicyCmdlineConfigMap host_policy_map_;
  std::string repoagent_dir_;
  RateLimitMode rate_limit_mode_;
  RateLimiter::ResourceMap rate_limit_resource_map_;

  // FIXME, remove once all backends use backend config.
  // Tensorflow options
  bool tf_soft_placement_enabled_;
  float tf_gpu_memory_fraction_;

  // Current state of the inference server.
  ServerReadyState ready_state_;

  // Number of in-flight, non-inference requests. During shutdown we
  // attempt to wait for all in-flight non-inference requests to
  // complete before exiting (also wait for in-flight inference
  // requests but that is determined by backend shared_ptr).
  std::atomic<uint64_t> inflight_request_counter_;

  std::shared_ptr<RateLimiter> rate_limiter_;
  std::unique_ptr<ModelRepositoryManager> model_repository_manager_;
  std::shared_ptr<PersistentBackendManager> persist_backend_manager_;
};

}}  // namespace nvidia::inferenceserver
