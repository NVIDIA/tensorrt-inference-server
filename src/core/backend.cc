// Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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
#include "src/core/logging.h"
#include "src/core/metric_model_reporter.h"
#include "src/core/model_config_utils.h"
#include "src/core/provider_utils.h"
#include "src/core/sequence_batch_scheduler.h"
#include "src/core/trtserver.h"

namespace nvidia { namespace inferenceserver {

Status
InferenceBackend::GetInput(
    const std::string& name, const ModelInput** input) const
{
  const auto itr = input_map_.find(name);
  if (itr == input_map_.end()) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "unexpected inference input '" + name + "' for model '" + Name() + "'");
  }

  *input = &itr->second;
  return Status::Success;
}

Status
InferenceBackend::GetOutput(
    const std::string& name, const ModelOutput** output) const
{
  const auto itr = output_map_.find(name);
  if (itr == output_map_.end()) {
    return Status(
        RequestStatusCode::INVALID_ARG, "unexpected inference output '" + name +
                                            "' for model '" + Name() + "'");
  }

  *output = &itr->second;
  return Status::Success;
}

Status
InferenceBackend::SetModelConfig(
    const std::string& path, const ModelConfig& config)
{
  config_ = config;
  RETURN_IF_ERROR(GetModelVersionFromPath(path, &version_));

  // Create the metric reporter for this backend.
  metric_reporter_ = std::make_shared<MetricModelReporter>(
      Name(), version_, config_.metric_tags());

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

  return Status::Success;
}

Status
InferenceBackend::SetScheduler(std::unique_ptr<Scheduler> scheduler)
{
  if (scheduler_ != nullptr) {
    return Status(
        RequestStatusCode::INTERNAL, "Attempt to change scheduler not allowed");
  }

  scheduler_ = std::move(scheduler);
  return Status::Success;
}

Status
InferenceBackend::SetConfiguredScheduler(
    const uint32_t runner_cnt, Scheduler::StandardInitFunc OnInit,
    Scheduler::StandardRunFunc OnRun)
{
  std::unique_ptr<Scheduler> scheduler;

  // Create a warmup function for the scheduler thread to run the contexts
  // in corresponding threads. Currently the warmup function can't be run
  // asynchronously with respect to Scheduler::Create() as there is no way to
  // change ModelReadyState, which is controlled by model manager, from within
  // the scheduler.
  // But running warmup synchronously allows us to use one set of warmup data
  // for all contexts.
  std::vector<WarmupData> samples;
  if (Config().model_warmup_size() != 0) {
    RETURN_IF_ERROR(GenerateWarmupData(&samples));
  }

  auto& model_name = Name();
  auto version = Version();
  auto OnWarmup = [this, &model_name, &version,
                   &samples](uint32_t runner_idx) -> Status {
    for (const auto& sample : samples) {
      std::vector<Scheduler::Payload> payloads;
      // Duplicate payloads to match batch size requirement.
      for (size_t idx = 0; idx < sample.batch_size_; idx++) {
        std::shared_ptr<InferRequestProvider> request_provider;
        RETURN_IF_ERROR(InferRequestProvider::Create(
            model_name, version, sample.request_header_, sample.input_buffer_,
            &request_provider));
        payloads.emplace_back(nullptr, request_provider, nullptr, nullptr);
      }

      std::promise<Status> warmup_promise;
      auto warmup_future = warmup_promise.get_future();
      Run(runner_idx, &payloads, [&warmup_promise](Status status) {
        warmup_promise.set_value(status);
      });
      RETURN_IF_ERROR(warmup_future.get());
    }
    return Status::Success;
  };

  // If 'sequence_batching' is configured use the SequenceBatchScheduler,
  // otherwise use the default DynamicBatchScheduler.
  if (config_.has_sequence_batching()) {
    RETURN_IF_ERROR(SequenceBatchScheduler::Create(
        config_, runner_cnt, OnInit, OnWarmup, OnRun, &scheduler));
  } else {
    RETURN_IF_ERROR(DynamicBatchScheduler::Create(
        config_, runner_cnt, OnInit, OnWarmup, OnRun, &scheduler));
  }

  return SetScheduler(std::move(scheduler));
}

Status
InferenceBackend::Init(
    const std::string& path, const ModelConfig& config,
    const std::string& platform)
{
  RETURN_IF_ERROR(ValidateModelConfig(config, platform));
  RETURN_IF_ERROR(SetModelConfig(path, config));

  return Status::Success;
}

void
InferenceBackend::Run(
    const std::shared_ptr<ModelInferStats>& stats,
    const std::shared_ptr<InferRequestProvider>& request_provider,
    const std::shared_ptr<InferResponseProvider>& response_provider,
    std::function<void(const Status&)> OnCompleteHandleInfer)
{
  scheduler_->Enqueue(
      stats, request_provider, response_provider, OnCompleteHandleInfer);
}

void
InferenceBackend::Run(
    uint32_t runner_idx, std::vector<Scheduler::Payload>* payloads,
    std::function<void(Status)> OnCompleteQueuedPayloads)
{
  // Each runner executes using the corresponding context...
  if (runner_idx >= contexts_.size()) {
    OnCompleteQueuedPayloads(Status(
        RequestStatusCode::INTERNAL,
        "unexpected runner index" + std::to_string(runner_idx) +
            ", max allowed " + std::to_string(contexts_.size())));
    return;
  }

  // Stop queue timer and start compute timer when the payload is
  // scheduled to run
  for (auto& payload : *payloads) {
    if (payload.stats_ != nullptr) {
      payload.stats_->CaptureTimestamp(
          ModelInferStats::TimestampKind::kComputeStart);
      payload.stats_->SetGPUDevice(contexts_[runner_idx]->gpu_device_);
    }
  }

  Status status = contexts_[runner_idx]->Run(this, payloads);

  // Stop compute timers.
  for (auto& payload : *payloads) {
    if (payload.stats_ != nullptr) {
      payload.stats_->CaptureTimestamp(
          ModelInferStats::TimestampKind::kComputeEnd);
    }
  }

  OnCompleteQueuedPayloads(status);
}

Status
InferenceBackend::GenerateWarmupData(std::vector<WarmupData>* samples)
{
  static std::string warmup_data_folder = "warmup";

  samples->clear();
  for (const auto& warmup_setting : config_.model_warmup()) {
    samples->emplace_back(warmup_setting.name(), warmup_setting.batch_size());
    auto& warmup_data = samples->back();

    // Two passes. First pass to get max byte size for synthetic data and
    // to generate request header
    size_t max_zero_byte_size = 0;
    size_t max_random_byte_size = 0;
    // use batch-1 for every request, batch size is simulated by populating
    // requests for single run.
    warmup_data.request_header_.set_batch_size(1);
    for (const auto& input_meta : warmup_setting.inputs()) {
      auto input = warmup_data.request_header_.add_input();
      input->set_name(input_meta.first);
      (*input->mutable_dims()) = input_meta.second.dims();

      auto batch_byte_size =
          GetByteSize(input_meta.second.data_type(), input_meta.second.dims());
      switch (input_meta.second.input_data_type_case()) {
        case ModelWarmup_InputMetaData::InputDataTypeCase::kZeroData: {
          if (batch_byte_size == -1) {
            batch_byte_size =
                GetElementCount(input_meta.second.dims()) * sizeof(int32_t);
          }
          max_zero_byte_size =
              std::max((size_t)batch_byte_size, max_zero_byte_size);
          break;
        }
        case ModelWarmup_InputMetaData::InputDataTypeCase::kRandomData: {
          if (batch_byte_size == -1) {
            batch_byte_size =
                GetElementCount(input_meta.second.dims()) * sizeof(int32_t);
            max_zero_byte_size =
                std::max((size_t)batch_byte_size, max_zero_byte_size);
          } else {
            max_random_byte_size =
                std::max((size_t)batch_byte_size, max_random_byte_size);
          }
          break;
        }
        case ModelWarmup_InputMetaData::InputDataTypeCase::kInputDataFile: {
          // For data provided from file, we can set buffer in first pass
          warmup_data.provided_data_.emplace_back();
          auto& input_data = warmup_data.provided_data_.back();
          RETURN_IF_ERROR(ReadTextFile(
              JoinPath({model_dir_, warmup_data_folder,
                        input_meta.second.input_data_file()}),
              &input_data));

          if (batch_byte_size == -1) {
            batch_byte_size = input_data.size();
          } else if (((size_t)batch_byte_size) > input_data.size()) {
            return Status(
                RequestStatusCode::INVALID_ARG,
                "warmup setting expects " + std::to_string(batch_byte_size) +
                    " bytes, but the data "
                    "provided from " +
                    input_meta.second.input_data_file() + "only has " +
                    std::to_string(input_data.size()) + " bytes");
          }

          auto pr = warmup_data.input_buffer_.emplace(input->name(), nullptr);
          pr.first->second.reset(new MemoryReference());
          static_cast<MemoryReference*>(pr.first->second.get())
              ->AddBuffer(
                  input_data.data(), input_data.size(),
                  TRTSERVER_MEMORY_CPU /* memory_type */,
                  0 /* memory_type_id */);
          break;
        }
        default:
          return Status(
              RequestStatusCode::INVALID_ARG,
              "warmup setting expects input '" + input_meta.first +
                  "' to have input_data_type set");
      }

      input->set_batch_byte_size(batch_byte_size);
    }
    for (const auto& io : Config().output()) {
      auto output = warmup_data.request_header_.add_output();
      output->set_name(io.name());
    }

    // Second pass to prepare input buffer if using synthetic data
    TRTSERVER_Memory_Type type;
    int64_t type_id;

    warmup_data.zero_data_.reset(new AllocatedSystemMemory(
        max_zero_byte_size, TRTSERVER_MEMORY_CPU /* memory_type */,
        0 /* memory_type_id */));
    char* zero_buffer = warmup_data.zero_data_->MutableBuffer(&type, &type_id);
    memset(zero_buffer, 0, max_zero_byte_size);

    warmup_data.random_data_.reset(new AllocatedSystemMemory(
        max_random_byte_size, TRTSERVER_MEMORY_CPU /* memory_type */,
        0 /* memory_type_id */));
    char* random_buffer =
        warmup_data.random_data_->MutableBuffer(&type, &type_id);

    for (const auto& input : warmup_data.request_header_.input()) {
      auto& input_meta = warmup_setting.inputs().find(input.name())->second;
      const auto input_data_type = input_meta.input_data_type_case();
      if (input_data_type ==
          ModelWarmup_InputMetaData::InputDataTypeCase::kInputDataFile) {
        continue;
      }

      auto allocated_buffer =
          ((input_data_type ==
            ModelWarmup_InputMetaData::InputDataTypeCase::kZeroData) ||
           (input_meta.data_type() == DataType::TYPE_STRING))
              ? zero_buffer
              : random_buffer;

      auto pr = warmup_data.input_buffer_.emplace(input.name(), nullptr);
      pr.first->second.reset(new MemoryReference());
      static_cast<MemoryReference*>(pr.first->second.get())
          ->AddBuffer(
              allocated_buffer, input.batch_byte_size(),
              TRTSERVER_MEMORY_CPU /* memory_type */, 0 /* memory_type_id */);
    }
  }

  return Status::Success;
}

}}  // namespace nvidia::inferenceserver
