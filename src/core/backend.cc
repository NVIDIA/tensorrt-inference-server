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

#include "src/core/backend.h"

#include <chrono>
#include <future>
#include "src/core/constants.h"
#include "src/core/dynamic_batch_scheduler.h"
#include "src/core/filesystem.h"
#include "src/core/infer_request.h"
#include "src/core/logging.h"
#include "src/core/model_config_utils.h"
#include "src/core/sequence_batch_scheduler.h"

namespace nvidia { namespace inferenceserver {

Status
InferenceBackend::GetInput(
    const std::string& name, const inference::ModelInput** input) const
{
  const auto itr = input_map_.find(name);
  if (itr == input_map_.end()) {
    return Status(
        Status::Code::INVALID_ARG,
        "unexpected inference input '" + name + "' for model '" + Name() + "'");
  }

  *input = &itr->second;
  return Status::Success;
}

Status
InferenceBackend::GetOutput(
    const std::string& name, const inference::ModelOutput** output) const
{
  const auto itr = output_map_.find(name);
  if (itr == output_map_.end()) {
    return Status(
        Status::Code::INVALID_ARG, "unexpected inference output '" + name +
                                       "' for model '" + Name() + "'");
  }

  *output = &itr->second;
  return Status::Success;
}

Status
InferenceBackend::SetModelConfig(
    const std::string& path, const inference::ModelConfig& config)
{
  config_ = config;
  RETURN_IF_ERROR(GetModelVersionFromPath(path, &version_));

  // Initialize the input map
  for (const auto& io : config.input()) {
    input_map_.insert(std::make_pair(io.name(), io));
  }

  // Initialize the output map and label provider for each output
  label_provider_ = std::make_shared<LabelProvider>();
  model_dir_ = DirName(path);
  for (const auto& io : config.output()) {
    output_map_.insert(std::make_pair(io.name(), io));

    if (!io.label_filename().empty()) {
      const auto label_path = JoinPath({model_dir_, io.label_filename()});
      RETURN_IF_ERROR(label_provider_->AddLabels(io.name(), label_path));
    }
  }

  if (config_.has_dynamic_batching()) {
    default_priority_level_ =
        config_.dynamic_batching().default_priority_level();
    max_priority_level_ = config_.dynamic_batching().priority_levels();
  } else if (config_.has_ensemble_scheduling()) {
    // For ensemble, allow any priority level to pass through
    default_priority_level_ = 0;
    max_priority_level_ = UINT32_MAX;
  } else {
    default_priority_level_ = 0;
    max_priority_level_ = 0;
  }

  return Status::Success;
}

Status
InferenceBackend::SetScheduler(std::unique_ptr<Scheduler> scheduler)
{
  if (scheduler_ != nullptr) {
    return Status(
        Status::Code::INTERNAL, "Attempt to change scheduler not allowed");
  }

  scheduler_ = std::move(scheduler);
  return Status::Success;
}

Status
InferenceBackend::SetConfiguredScheduler(void* model)
{
  std::unique_ptr<Scheduler> scheduler;

  // Need to enforce equal shape batches (i.e. non-ragged batches) if
  // the model 1) allows one or more variable-size input tensors that
  // are not marked as 'allow_ragged_batch' or 2) has one or more
  // shape-tensor inputs. This is not needed if all input shapes are
  // non-variable and if there are no shape tensors... so we don't
  // enable it in that case for efficiency reasons.
  std::unordered_map<std::string, bool> enforce_equal_shape_tensors;
  for (const auto input : config_.input()) {
    if (input.is_shape_tensor()) {
      enforce_equal_shape_tensors.insert({input.name(), true});
    } else if (!input.allow_ragged_batch() && (GetElementCount(input) == -1)) {
      enforce_equal_shape_tensors.insert({input.name(), false});
    }
  }

  // If 'sequence_batching' is configured use the SequenceBatchScheduler,
  // otherwise use the default DynamicBatchScheduler.
  if (config_.has_sequence_batching()) {
    // Sequence batcher
    RETURN_IF_ERROR(SequenceBatchScheduler::Create(
        static_cast<TritonModel*>(model), enforce_equal_shape_tensors,
        &scheduler));
  } else if (config_.has_dynamic_batching()) {
    // Dynamic batcher
    RETURN_IF_ERROR(DynamicBatchScheduler::Create(
        static_cast<TritonModel*>(model), nullptr, 0 /*nice*/,
        true /* dynamic_batching_enabled */, config_.max_batch_size(),
        enforce_equal_shape_tensors, config_.dynamic_batching(), &scheduler));
  } else {
    // Default scheduler. Use dynamic batch scheduler (with batching
    // disabled) as the default scheduler.
    RETURN_IF_ERROR(DynamicBatchScheduler::Create(
        static_cast<TritonModel*>(model), nullptr, 0 /*nice*/,
        false /* dynamic_batching_enabled */, 1 /* max_batch_size */,
        std::unordered_map<
            std::string, bool>() /* enforce_equal_shape_tensors */,
        false /* preserve_ordering */,
        std::set<int32_t>() /* preferred_batch_sizes */,
        0 /* max_queue_delay_microseconds */, &scheduler));
  }

  return SetScheduler(std::move(scheduler));
}

Status
InferenceBackend::Init(
    const std::string& path, const inference::ModelConfig& config,
    const std::string& platform)
{
  RETURN_IF_ERROR(
      ValidateModelConfig(config, platform, min_compute_capability_));
  RETURN_IF_ERROR(ValidateModelIOConfig(config));
  RETURN_IF_ERROR(SetModelConfig(path, config));

  return Status::Success;
}

void
InferenceBackend::Run(
    uint32_t runner_idx,
    std::vector<std::unique_ptr<InferenceRequest>>&& requests)
{
  // Each runner executes using the corresponding context...  If the
  // runner_idx is invalid then the scheduler has done something badly
  // wrong so fail and release all requests.
  if (runner_idx >= contexts_.size()) {
    InferenceRequest::RespondIfError(
        requests,
        Status(
            Status::Code::INTERNAL,
            "unexpected runner index" + std::to_string(runner_idx) +
                ", max allowed " + std::to_string(contexts_.size())),
        true /*release_requests */);
    return;
  }

  contexts_[runner_idx]->Run(this, std::move(requests));
}

}}  // namespace nvidia::inferenceserver
