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

#include "model_config.pb.h"
#include "src/core/backend_context.h"
#include "src/core/infer_stats.h"
#include "src/core/label_provider.h"
#include "src/core/scheduler.h"
#include "src/core/status.h"

namespace nvidia { namespace inferenceserver {

#ifdef TRITON_ENABLE_STATS
#define FAIL_ALL_AND_RETURN_IF_ERROR(REQUESTS, RESPONSES, MR, S, LOG_MSG)    \
  do {                                                                       \
    const auto& status__ = (S);                                              \
    if (!status__.IsOk()) {                                                  \
      for (auto& response : (RESPONSES)) {                                   \
        if (response != nullptr) {                                           \
          const auto& response_status__ = InferenceResponse::SendWithStatus( \
              std::move(response), TRITONSERVER_RESPONSE_COMPLETE_FINAL,     \
              status__);                                                     \
          LOG_STATUS_ERROR(response_status__, (LOG_MSG));                    \
        }                                                                    \
      }                                                                      \
      for (auto& request : (REQUESTS)) {                                     \
        request->ReportStatistics(MR, false /* success */, 0, 0, 0, 0);      \
        InferenceRequest::Release(                                           \
            std::move(request), TRITONSERVER_REQUEST_RELEASE_ALL);           \
      }                                                                      \
      return;                                                                \
    }                                                                        \
  } while (false)
#else
#define FAIL_ALL_AND_RETURN_IF_ERROR(REQUESTS, RESPONSES, MR, S, LOG_MSG)    \
  do {                                                                       \
    const auto& status__ = (S);                                              \
    if (!status__.IsOk()) {                                                  \
      for (auto& response : (RESPONSES)) {                                   \
        if (response != nullptr) {                                           \
          const auto& response_status__ = InferenceResponse::SendWithStatus( \
              std::move(response), TRITONSERVER_RESPONSE_COMPLETE_FINAL,     \
              status__);                                                     \
          LOG_STATUS_ERROR(response_status__, (LOG_MSG));                    \
        }                                                                    \
      }                                                                      \
      for (auto& request : (REQUESTS)) {                                     \
        InferenceRequest::Release(                                           \
            std::move(request), TRITONSERVER_REQUEST_RELEASE_ALL);           \
      }                                                                      \
      return;                                                                \
    }                                                                        \
  } while (false)
#endif  // TRITON_ENABLE_STATS

class InferenceRequest;

//
// Interface for backends that handle inference requests.
//
class InferenceBackend {
 public:
  explicit InferenceBackend(const double min_compute_capability)
      : min_compute_capability_(min_compute_capability)
  {
  }
  virtual ~InferenceBackend() {}

  // Get the name of model being served.
  const std::string& Name() const { return config_.name(); }

  // Get the version of model being served.
  int64_t Version() const { return version_; }

  // Get the configuration of model being served.
  const inference::ModelConfig& Config() const { return config_; }

  // Get the stats collector for the model being served.
  InferenceStatsAggregator* MutableStatsAggregator()
  {
    return &stats_aggregator_;
  }
  const InferenceStatsAggregator& StatsAggregator() const
  {
    return stats_aggregator_;
  }

  // Get the model configuration for a named input.
  Status GetInput(
      const std::string& name, const inference::ModelInput** input) const;

  // Get the model configuration for a named output.
  Status GetOutput(
      const std::string& name, const inference::ModelOutput** output) const;

  // Get a label provider for the model.
  const std::shared_ptr<LabelProvider>& GetLabelProvider() const
  {
    return label_provider_;
  }

  Status Init(
      const std::string& path, const inference::ModelConfig& config,
      const std::string& platform);

  // Enqueue a request for execution. If Status::Success is returned
  // then the backend has taken ownership of the request object and so
  // 'request' will be nullptr. If non-success is returned then the
  // caller still retains ownership of 'request'.
  Status Enqueue(std::unique_ptr<InferenceRequest>& request)
  {
    return scheduler_->Enqueue(request);
  }

  uint32_t DefaultPriorityLevel() const { return default_priority_level_; }

  uint32_t MaxPriorityLevel() const { return max_priority_level_; }

  bool DecoupledTransactionPolicy()
  {
    return config_.model_transaction_policy().decoupled();
  }

 protected:
  // Run model on the context associated with 'runner_idx' to execute
  // for one or more requests. This function takes ownership of
  // 'requests' and is responsible for generating responses and
  // releasing the requests.
  virtual void Run(
      uint32_t runner_idx,
      std::vector<std::unique_ptr<InferenceRequest>>&& requests);

  // Set the configuration of the model being served.
  Status SetModelConfig(
      const std::string& path, const inference::ModelConfig& config);

  // Explicitly set the scheduler to use for inference requests to the
  // model. The scheduler can only be set once for a backend.
  Status SetScheduler(std::unique_ptr<Scheduler> scheduler);

  // Set the scheduler based on the model configuration. The scheduler
  // can only be set once for a backend.
  // FIXME: The pointer is of TritonModel* type and is doen to keep ensemble
  // happy
  Status SetConfiguredScheduler(void* model);

  // Get the raw pointer to the scheduler of this backend.
  Scheduler* BackendScheduler() { return scheduler_.get(); }

  std::vector<std::unique_ptr<BackendContext>> contexts_;

  // The scheduler to use for this backend.
  std::unique_ptr<Scheduler> scheduler_;

 private:
  // The minimum supported CUDA compute capability.
  const double min_compute_capability_;

  // Configuration of the model that this backend represents.
  inference::ModelConfig config_;

  // Version of the model that this backend represents.
  int64_t version_;

  // The stats collector for the model that this backend represents.
  InferenceStatsAggregator stats_aggregator_;

  // Label provider for this model.
  std::shared_ptr<LabelProvider> label_provider_;

  // Map from input name to the model configuration for that input.
  std::unordered_map<std::string, inference::ModelInput> input_map_;

  // Map from output name to the model configuration for that output.
  std::unordered_map<std::string, inference::ModelOutput> output_map_;

  // Path to model
  std::string model_dir_;

  // The default priority level for the backend.
  uint32_t default_priority_level_;

  // The largest priority value for the backend.
  uint32_t max_priority_level_;
};

}}  // namespace nvidia::inferenceserver
