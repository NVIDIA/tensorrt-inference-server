// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include <memory>
#include <string>
#include "model_config.pb.h"
#include "src/backends/backend/triton_backend_manager.h"
#include "src/core/backend.h"
#include "src/core/filesystem.h"
#include "src/core/infer_request.h"
#include "src/core/status.h"

namespace nvidia { namespace inferenceserver {

class InferenceServer;
class TritonModelInstance;

//
// Represents a model.
//
// Inheriting from InferenceBackend (which is misnamed) required for
// now to interface with legacy arch.
//
class TritonModel : public InferenceBackend {
 public:
  static Status Create(
      InferenceServer* server, const std::string& model_repository_path,
      const BackendCmdlineConfigMap& backend_cmdline_config_map,
      const HostPolicyCmdlineConfigMap& host_policy_map,
      const std::string& model_name, const int64_t version,
      const inference::ModelConfig& model_config,
      std::unique_ptr<TritonModel>* model);
  ~TritonModel();

  const std::string& LocalizedModelPath() const
  {
    return localized_model_dir_->Path();
  }
  InferenceServer* Server() { return server_; }
  bool AutoCompleteConfig() const { return auto_complete_config_; }
  Status UpdateModelConfig(
      const uint32_t config_version,
      TRITONSERVER_Message* updated_config_message);
  const std::shared_ptr<TritonBackend>& Backend() const { return backend_; }
  const std::vector<std::unique_ptr<TritonModelInstance>>& Instances() const
  {
    return instances_;
  }
  void* State() { return state_; }
  void SetState(void* state) { state_ = state; }
  Status AddInstance(
      std::unique_ptr<TritonModelInstance>&& instance, const bool passive,
      const inference::ModelRateLimiter& rate_limiter_config);

 private:
  DISALLOW_COPY_AND_ASSIGN(TritonModel);

  TritonModel(
      InferenceServer* server,
      const std::shared_ptr<LocalizedDirectory>& localized_model_dir,
      const std::shared_ptr<TritonBackend>& backend,
      const double min_compute_capability, const bool auto_complete_config);

  Status Initialize();
  Status WarmUp();

  // The server object that owns this model. The model holds this as a
  // raw pointer because the lifetime of the server is guaranteed to
  // be longer than the lifetime of a model owned by the server.
  InferenceServer* server_;

  // Whether the backend should attempt to auto-complete the model config.
  const bool auto_complete_config_;

  // The localized repo directory holding the model. If localization
  // required creation of a temporary local copy then that copy will
  // persist as along as this object is retained by this model.
  std::shared_ptr<LocalizedDirectory> localized_model_dir_;

  // Backend used by this model.
  std::shared_ptr<TritonBackend> backend_;

  // The model instances for this model.
  std::vector<std::unique_ptr<TritonModelInstance>> instances_;
  std::vector<std::unique_ptr<TritonModelInstance>> passive_instances_;

  // Opaque state associated with this model.
  void* state_;

  // Whether the model is initialized.
  bool initialized_;
};

}}  // namespace nvidia::inferenceserver
