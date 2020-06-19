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

#include "src/core/infer_request.h"

#include <deque>
#include "src/core/backend.h"
#include "src/core/logging.h"
#include "src/core/server.h"

namespace nvidia { namespace inferenceserver {

namespace {

// Utilities for Null request feature.
TRITONSERVER_Error*
NullResponseAlloc(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
    int64_t preferred_memory_type_id, void* userp, void** buffer,
    void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type,
    int64_t* actual_memory_type_id)
{
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_INTERNAL,
      "unexpected allocation for null request, no output should be requestd.");
}

TRITONSERVER_Error*
NullResponseRelease(
    TRITONSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
    size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id)
{
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_INTERNAL,
      "unexpected release for null request, no output should be requestd.");
}

ResponseAllocator null_allocator =
    ResponseAllocator(NullResponseAlloc, NullResponseRelease);

void
NullResponseComplete(TRITONSERVER_InferenceResponse* iresponse, void* userp)
{
  LOG_TRITONSERVER_ERROR(
      TRITONSERVER_InferenceResponseDelete(iresponse),
      "deleting null response");
}

void
NullRequestComplete(
    TRITONSERVER_InferenceRequest* request, const uint32_t flags, void* userp)
{
  if ((flags & TRITONSERVER_REQUEST_RELEASE_ALL) != 0) {
    LOG_TRITONSERVER_ERROR(
        TRITONSERVER_InferenceRequestDelete(request), "deleting null request");
  }
}

}  // namespace

const std::string&
InferenceRequest::ModelName() const
{
  return backend_raw_->Name();
}

int64_t
InferenceRequest::ActualModelVersion() const
{
  return backend_raw_->Version();
}

void
InferenceRequest::SetPriority(uint32_t p)
{
  if ((p == 0) || (p > backend_raw_->MaxPriorityLevel())) {
    priority_ = backend_raw_->DefaultPriorityLevel();
  } else {
    priority_ = p;
  }
}

Status
InferenceRequest::Run(std::unique_ptr<InferenceRequest>& request)
{
  return request->backend_raw_->Enqueue(request);
}

void
InferenceRequest::RespondIfError(
    std::unique_ptr<InferenceRequest>& request, const Status& status,
    const bool release_request)
{
  if (status.IsOk()) {
    return;
  }

  // Use the response factory to create a response, set the status,
  // and send it. If something goes wrong all we can do is log the
  // error.
  std::unique_ptr<InferenceResponse> response;
  LOG_STATUS_ERROR(
      request->response_factory_.CreateResponse(&response),
      "failed to create error response");
  LOG_STATUS_ERROR(
      InferenceResponse::SendWithStatus(std::move(response), status),
      "failed to send error response");

  // If releasing the request then invoke the release callback which
  // gives ownership to the callback. So can't access 'request' after
  // this point.
  if (release_request) {
    InferenceRequest::Release(
        std::move(request), TRITONSERVER_REQUEST_RELEASE_ALL);
  }
}

void
InferenceRequest::RespondIfError(
    std::vector<std::unique_ptr<InferenceRequest>>& requests,
    const Status& status, const bool release_requests)
{
  if (status.IsOk()) {
    return;
  }

  for (auto& request : requests) {
    RespondIfError(request, status, release_requests);
  }
}

void
InferenceRequest::Release(
    std::unique_ptr<InferenceRequest>&& request, const uint32_t release_flags)
{
  // Invoke the release callbacks added internally before releasing the
  // request to user provided callback.
  for (auto it = request->release_callbacks_.rbegin();
       it != request->release_callbacks_.rend(); it++) {
    (*it)();
  }
  request->release_callbacks_.clear();

#ifdef TRITON_ENABLE_TRACING
  // Take ownership of trace object so it is not lost when we release
  // the request below.
  std::unique_ptr<InferenceTrace> trace = std::move(request->trace_);
#endif  // TRITON_ENABLE_TRACING

  void* userp = request->release_userp_;
  request->release_fn_(
      reinterpret_cast<TRITONSERVER_InferenceRequest*>(request.release()),
      release_flags, userp);

#ifdef TRITON_ENABLE_TRACING
  // If tracing then record request end (after the callback completes
  // so that any callback overhead is included in the request time)
  // and release the trace.
  if (trace != nullptr) {
    trace->ReportNow(TRITONSERVER_TRACE_REQUEST_END);
    InferenceTrace::Release(std::move(trace));
  }
#endif  // TRITON_ENABLE_TRACING
}

InferenceRequest*
InferenceRequest::CopyAsNull(const InferenceRequest& from)
{
  // Create a copy of 'from' request with artifical inputs and no requested
  // outputs. Maybe more efficient to share inputs and other metadata,
  // but that binds the Null request with 'from' request's lifecycle.
  std::unique_ptr<InferenceRequest> lrequest(
      new InferenceRequest(from.backend_raw_, from.requested_model_version_));
  lrequest->needs_normalization_ = false;
  lrequest->batch_size_ = from.batch_size_;
  lrequest->collect_stats_ = false;

  // Three passes: first to construct input for the shape tensors inputs, second
  // to obtain the max input byte size for allocating a large enough buffer for
  // all non shape tensor inputs; third to construct the inputs for these
  // tensors.
  //  First pass
  for (const auto& input : from.OriginalInputs()) {
    // Handle only shape tensors in this pass
    if (!input.second.IsShapeTensor()) {
      continue;
    }

    // Prepare the memory to hold input data
    size_t byte_size = input.second.Data()->TotalByteSize();
    auto mem_type = TRITONSERVER_MEMORY_CPU;
    int64_t mem_id = 0;
    std::shared_ptr<MutableMemory> data =
        std::make_shared<AllocatedMemory>(byte_size, mem_type, mem_id);

    // Get the source buffer. Assumes shape tensors be in a single buffer on the
    // CPU
    const auto& from_data = input.second.Data();
    size_t from_data_byte_size;
    TRITONSERVER_MemoryType from_data_memory_type;
    int64_t from_data_memory_id;
    const char* from_data_buffer = from_data->BufferAt(
        0 /* idx */, &from_data_byte_size, &from_data_memory_type,
        &from_data_memory_id);

    if (from_data_byte_size != byte_size) {
      LOG_WARNING
          << "The byte size of shape tensor to be copied does not match";
    }

    // Copy the shape values to the input buffer
    std::memcpy(data->MutableBuffer(), from_data_buffer, from_data_byte_size);

    Input* new_input;
    lrequest->AddOriginalInput(
        input.first, input.second.DType(), input.second.Shape(), &new_input);

    // Must normalize shape here...
    *new_input->MutableShape() = new_input->OriginalShape();
    *new_input->MutableShapeWithBatchDim() = new_input->OriginalShape();

    new_input->SetData(data);
  }


  // Second pass
  size_t max_byte_size = 0;
  const std::string* max_input_name;
  for (const auto& input : from.OriginalInputs()) {
    // Skip shape tensors in this pass
    if (input.second.IsShapeTensor()) {
      continue;
    }
    if (input.second.Data()->TotalByteSize() >= max_byte_size) {
      max_byte_size = input.second.Data()->TotalByteSize();
      max_input_name = &(input.first);
    }
  }

  // Third pass
  // [DLIS-1268] should use one growable static buffer for all null requests
  auto mem_type = TRITONSERVER_MEMORY_CPU;
  int64_t mem_id = 0;
  std::shared_ptr<Memory> data =
      std::make_shared<AllocatedMemory>(max_byte_size, mem_type, mem_id);
  auto data_base = data->BufferAt(0, &max_byte_size, &mem_type, &mem_id);
  for (const auto& input : from.OriginalInputs()) {
    // skip shape tensors in this pass
    if (input.second.IsShapeTensor()) {
      continue;
    }
    Input* new_input;
    lrequest->AddOriginalInput(
        input.first, input.second.DType(), input.second.Shape(), &new_input);

    // Must normalize shape here...
    *new_input->MutableShape() = new_input->OriginalShape();
    *new_input->MutableShapeWithBatchDim() = new_input->OriginalShape();

    // Note that the input that have max byte size will be responsible for
    // holding the artifical data, while other inputs will hold a reference to
    // it with byte size that matches 'from'
    if (input.first == *max_input_name) {
      new_input->SetData(data);
    } else {
      new_input->AppendData(
          data_base, input.second.Data()->TotalByteSize(), mem_type, mem_id);
    }
  }

  // No outputs were requested and thus there should be no allocations.
  lrequest->SetResponseCallback(
      &null_allocator, nullptr, NullResponseComplete, nullptr);
  lrequest->SetReleaseCallback(NullRequestComplete, nullptr);

  // Must normalize inputs here...
  for (auto& pr : lrequest->original_inputs_) {
    lrequest->inputs_.emplace(
        std::make_pair(pr.first, std::addressof(pr.second)));
  }

  return lrequest.release();
}

Status
InferenceRequest::MutableOriginalInput(
    const std::string& name, InferenceRequest::Input** input)
{
  auto itr = original_inputs_.find(name);
  if (itr == original_inputs_.end()) {
    return Status(
        Status::Code::INVALID_ARG,
        "input '" + name + "' does not exist in request");
  }

  *input = &(itr->second);

  return Status::Success;
}

Status
InferenceRequest::ImmutableInput(
    const std::string& name, const InferenceRequest::Input** input) const
{
  auto itr = inputs_.find(name);
  if (itr == inputs_.end()) {
    return Status(
        Status::Code::INVALID_ARG,
        "input '" + name + "' does not exist in request");
  }

  *input = itr->second;
  return Status::Success;
}

Status
InferenceRequest::AddOriginalInput(
    const std::string& name, const DataType datatype, const int64_t* shape,
    const uint64_t dim_count, InferenceRequest::Input** input)
{
  const auto& pr = original_inputs_.emplace(
      std::piecewise_construct, std::forward_as_tuple(name),
      std::forward_as_tuple(name, datatype, shape, dim_count));
  if (!pr.second) {
    return Status(
        Status::Code::INVALID_ARG,
        "input '" + name + "' already exists in request");
  }

  LOG_VERBOSE(1) << "add original input: " << *this;

  if (input != nullptr) {
    *input = std::addressof(pr.first->second);
  }

  needs_normalization_ = true;
  return Status::Success;
}

Status
InferenceRequest::AddOriginalInput(
    const std::string& name, const DataType datatype,
    const std::vector<int64_t>& shape, InferenceRequest::Input** input)
{
  return AddOriginalInput(name, datatype, &shape[0], shape.size(), input);
}

Status
InferenceRequest::RemoveOriginalInput(const std::string& name)
{
  if (original_inputs_.erase(name) != 1) {
    return Status(
        Status::Code::INVALID_ARG,
        "input '" + name + "' does not exist in request");
  }

  needs_normalization_ = true;
  return Status::Success;
}

Status
InferenceRequest::RemoveAllOriginalInputs()
{
  original_inputs_.clear();
  needs_normalization_ = true;
  return Status::Success;
}

Status
InferenceRequest::AddOverrideInput(
    const std::string& name, const DataType datatype,
    const std::vector<int64_t>& shape,
    std::shared_ptr<InferenceRequest::Input>* input)
{
  std::shared_ptr<Input> i = std::make_shared<Input>(name, datatype, shape);
  *(i->MutableShape()) = i->OriginalShape();
  *(i->MutableShapeWithBatchDim()) = i->OriginalShape();

  RETURN_IF_ERROR(AddOverrideInput(i));
  if (input != nullptr) {
    *input = std::move(i);
  }

  return Status::Success;
}

Status
InferenceRequest::AddOverrideInput(
    const std::shared_ptr<InferenceRequest::Input>& input)
{
  LOG_VERBOSE(1) << "adding input override for " << input->Name() << ": "
                 << *this;

  const auto& pr =
      override_inputs_.emplace(std::make_pair(input->Name(), input));
  if (!pr.second) {
    pr.first->second = input;
  }

  // Add or replace this override in the inputs...
  const auto res = inputs_.emplace(std::make_pair(input->Name(), input.get()));
  if (!res.second) {
    res.first->second = input.get();
  }

  LOG_VERBOSE(1) << "added input override for " << input->Name() << ": "
                 << *this;

  return Status::Success;
}

Status
InferenceRequest::AddOriginalRequestedOutput(const std::string& name)
{
  original_requested_outputs_.insert(name);
  needs_normalization_ = true;
  return Status::Success;
}

Status
InferenceRequest::RemoveOriginalRequestedOutput(const std::string& name)
{
  original_requested_outputs_.erase(name);
  needs_normalization_ = true;
  return Status::Success;
}

Status
InferenceRequest::RemoveAllOriginalRequestedOutputs()
{
  original_requested_outputs_.clear();
  needs_normalization_ = true;
  return Status::Success;
}

Status
InferenceRequest::PrepareForInference()
{
  // Remove override inputs as those are added during any previous
  // inference execution.
  inputs_.clear();
  override_inputs_.clear();

  // Renormalize if anything has changed in the inference request in a
  // way that could impact renormalization.
  if (needs_normalization_) {
    RETURN_IF_ERROR(Normalize());
    needs_normalization_ = false;
  }

  // Initially show the actual inputs to be only the original
  // inputs. If overrides are added later they will be added to
  // 'inputs_'.
  for (auto& pr : original_inputs_) {
    inputs_.emplace(std::make_pair(pr.first, std::addressof(pr.second)));
  }

  // Clear the timestamps
  queue_start_ns_ = 0;
#ifdef TRITON_ENABLE_STATS
  request_start_ns_ = 0;
#endif  // TRITON_ENABLE_STATS

  LOG_VERBOSE(1) << "prepared: " << *this;

  return Status::Success;
}

Status
InferenceRequest::Normalize()
{
  const ModelConfig& model_config = backend_raw_->Config();

  // Initialize the requested outputs to be used during inference. If
  // original_requested_outputs_ is empty assume all outputs specified
  // in model config are being requested.
  requested_outputs_.clear();
  if (original_requested_outputs_.size() == 0) {
    for (const auto& output : model_config.output()) {
      requested_outputs_.insert(output.name());
    }
  } else {
    // Validate if the original requested output name exists in the
    // model configuration.
    for (const auto& output_name : original_requested_outputs_) {
      const ModelOutput* output_config;
      RETURN_IF_ERROR(backend_raw_->GetOutput(output_name, &output_config));
    }
  }

  // Make sure that the request is providing the same number of inputs
  // as is expected by the model.
  if (original_inputs_.size() != (size_t)model_config.input_size()) {
    return Status(
        Status::Code::INVALID_ARG,
        "expected " + std::to_string(model_config.input_size()) +
            " inputs but got " + std::to_string(original_inputs_.size()) +
            " inputs for model '" + ModelName() + "'");
  }

  // Determine the batch size and shape of each input.
  if (model_config.max_batch_size() == 0) {
    // Model does not support Triton-style batching so set as
    // batch-size 0 and leave the tensor shapes as they are.
    batch_size_ = 0;
    for (auto& pr : original_inputs_) {
      auto& input = pr.second;
      *input.MutableShape() = input.OriginalShape();
    }
  } else {
    // Model does support Triton-style batching so each input tensor
    // must have the same first dimension which is the batch
    // size. Adjust the shape of the input tensors to remove the batch
    // dimension.
    batch_size_ = 0;
    for (auto& pr : original_inputs_) {
      auto& input = pr.second;

      // For a shape tensor, keep the tensor's shape as it is and mark
      // that the input is a shape tensor.
      const ModelInput* input_config;
      RETURN_IF_ERROR(backend_raw_->GetInput(pr.first, &input_config));
      if (input_config->is_shape_tensor()) {
        *input.MutableShape() = input.OriginalShape();
        input.SetIsShapeTensor(true);
        continue;
      }

      if (input.OriginalShape().size() == 0) {
        return Status(
            Status::Code::INVALID_ARG,
            "input '" + input.Name() +
                "' has no shape but model requires batch dimension for '" +
                ModelName() + "'");
      }

      if (batch_size_ == 0) {
        batch_size_ = input.OriginalShape()[0];
      } else if (input.OriginalShape()[0] != batch_size_) {
        return Status(
            Status::Code::INVALID_ARG,
            "input '" + input.Name() +
                "' batch size does not match other inputs for '" + ModelName() +
                "'");
      }

      input.MutableShape()->assign(
          input.OriginalShape().begin() + 1, input.OriginalShape().end());
    }
  }

  // Make sure request batch-size doesn't exceed what is supported by
  // the model.
  if ((int)batch_size_ > model_config.max_batch_size()) {
    return Status(
        Status::Code::INVALID_ARG,
        "inference request batch-size must be <= " +
            std::to_string(model_config.max_batch_size()) + " for '" +
            ModelName() + "'");
  }

  // Verify that each input shape is valid for the model, make
  // adjustments for reshapes and find the total tensor size.
  for (auto& pr : original_inputs_) {
    const ModelInput* input_config;
    RETURN_IF_ERROR(backend_raw_->GetInput(pr.first, &input_config));

    auto& input = pr.second;
    auto shape = input.MutableShape();

    if (input.DType() != input_config->data_type()) {
      return Status(
          Status::Code::INVALID_ARG,
          "inference input data-type is '" +
              std::string(DataTypeToProtocolString(input.DType())) +
              "', model expects '" +
              std::string(DataTypeToProtocolString(input_config->data_type())) +
              "' for '" + ModelName() + "'");
    }

    if (!CompareDimsWithWildcard(input_config->dims(), *shape)) {
      DimsList full_dims;
      if (model_config.max_batch_size() > 0) {
        full_dims.Add(WILDCARD_DIM);
      }
      for (int i = 0; i < input_config->dims_size(); ++i) {
        full_dims.Add(input_config->dims(i));
      }
      return Status(
          Status::Code::INVALID_ARG,
          "unexpected shape for input '" + pr.first + "' for model '" +
              ModelName() + "'. Expected " + DimsListToString(full_dims) +
              ", got " + DimsListToString(input.OriginalShape()));
    }

    // If there is a reshape for this input then adjust them to
    // match the reshape. As reshape may have variable-size
    // dimensions, we need to record corresponding value so that we
    // can set the value correctly for reshape.
    if (input_config->has_reshape()) {
      std::deque<int64_t> variable_size_values;
      for (int64_t idx = 0; idx < input_config->dims_size(); idx++) {
        if (input_config->dims(idx) == -1) {
          variable_size_values.push_back((*shape)[idx]);
        }
      }

      shape->clear();
      for (const auto& dim : input_config->reshape().shape()) {
        if (dim == -1) {
          shape->push_back(variable_size_values.front());
          variable_size_values.pop_front();
        } else {
          shape->push_back(dim);
        }
      }
    }

    // Create shape with batch dimension.
    // FIXME, should not need this!!
    if (batch_size_ == 0) {
      *input.MutableShapeWithBatchDim() = *shape;
    } else {
      input.MutableShapeWithBatchDim()->clear();
      input.MutableShapeWithBatchDim()->push_back(batch_size_);
      for (int64_t d : *shape) {
        input.MutableShapeWithBatchDim()->push_back(d);
      }
    }
  }

  return Status::Success;
}

#ifdef TRITON_ENABLE_STATS
void
InferenceRequest::ReportStatistics(
    MetricModelReporter* metric_reporter, bool success,
    const uint64_t compute_start_ns, const uint64_t compute_input_end_ns,
    const uint64_t compute_output_start_ns, const uint64_t compute_end_ns)
{
  if (!collect_stats_) {
    return;
  }

  INFER_STATS_DECL_TIMESTAMP(request_end_ns);

  if (success) {
    backend_raw_->MutableStatsAggregator()->UpdateSuccess(
        metric_reporter, std::max(1U, batch_size_), request_start_ns_,
        queue_start_ns_, compute_start_ns, compute_input_end_ns,
        compute_output_start_ns, compute_end_ns, request_end_ns);
    if (secondary_stats_aggregator_ != nullptr) {
      secondary_stats_aggregator_->UpdateSuccess(
          nullptr /* metric_reporter */, std::max(1U, batch_size_),
          request_start_ns_, queue_start_ns_, compute_start_ns,
          compute_input_end_ns, compute_output_start_ns, compute_end_ns,
          request_end_ns);
    }
  } else {
    backend_raw_->MutableStatsAggregator()->UpdateFailure(
        metric_reporter, request_start_ns_, request_end_ns);
    if (secondary_stats_aggregator_ != nullptr) {
      secondary_stats_aggregator_->UpdateFailure(
          nullptr /* metric_reporter */, request_start_ns_, request_end_ns);
    }
  }
}

void
InferenceRequest::ReportStatisticsWithDuration(
    MetricModelReporter* metric_reporter, bool success,
    const uint64_t compute_start_ns, const uint64_t compute_input_duration_ns,
    const uint64_t compute_infer_duration_ns,
    const uint64_t compute_output_duration_ns)
{
  if (!collect_stats_) {
    return;
  }

  INFER_STATS_DECL_TIMESTAMP(request_end_ns);

  if (success) {
    backend_raw_->MutableStatsAggregator()->UpdateSuccess(
        metric_reporter, std::max(1U, batch_size_), request_start_ns_,
        queue_start_ns_, compute_start_ns, request_end_ns,
        compute_input_duration_ns, compute_infer_duration_ns,
        compute_output_duration_ns);
    if (secondary_stats_aggregator_ != nullptr) {
      secondary_stats_aggregator_->UpdateSuccess(
          nullptr /* metric_reporter */, std::max(1U, batch_size_),
          request_start_ns_, queue_start_ns_, compute_start_ns, request_end_ns,
          compute_input_duration_ns, compute_infer_duration_ns,
          compute_output_duration_ns);
    }
  } else {
    backend_raw_->MutableStatsAggregator()->UpdateFailure(
        metric_reporter, request_start_ns_, request_end_ns);
    if (secondary_stats_aggregator_ != nullptr) {
      secondary_stats_aggregator_->UpdateFailure(
          nullptr /* metric_reporter */, request_start_ns_, request_end_ns);
    }
  }
}
#endif  // TRITON_ENABLE_STATS

//
// Input
//
InferenceRequest::Input::Input() : data_(new MemoryReference) {}

InferenceRequest::Input::Input(
    const std::string& name, const DataType datatype, const int64_t* shape,
    const uint64_t dim_count)
    : name_(name), datatype_(datatype),
      original_shape_(shape, shape + dim_count), is_shape_tensor_(false),
      data_(new MemoryReference)
{
}

InferenceRequest::Input::Input(
    const std::string& name, const DataType datatype,
    const std::vector<int64_t>& shape)
    : name_(name), datatype_(datatype), original_shape_(shape),
      is_shape_tensor_(false), data_(new MemoryReference)
{
}

Status
InferenceRequest::Input::SetIsShapeTensor(const bool is_shape_tensor)
{
  is_shape_tensor_ = is_shape_tensor;
  return Status::Success;
}

Status
InferenceRequest::Input::AppendData(
    const void* base, size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id)
{
  if (byte_size > 0) {
    std::static_pointer_cast<MemoryReference>(data_)->AddBuffer(
        static_cast<const char*>(base), byte_size, memory_type, memory_type_id);
  }

  return Status::Success;
}

Status
InferenceRequest::Input::SetData(const std::shared_ptr<Memory>& data)
{
  if (data_->TotalByteSize() != 0) {
    return Status(
        Status::Code::INVALID_ARG,
        "input '" + name_ + "' already has data, can't overwrite");
  }

  data_ = data;

  return Status::Success;
}

Status
InferenceRequest::Input::RemoveAllData()
{
  data_ = std::make_shared<MemoryReference>();
  return Status::Success;
}

Status
InferenceRequest::Input::DataBuffer(
    const size_t idx, const void** base, size_t* byte_size,
    TRITONSERVER_MemoryType* memory_type, int64_t* memory_type_id) const
{
  *base = data_->BufferAt(idx, byte_size, memory_type, memory_type_id);

  return Status::Success;
}


std::ostream&
operator<<(std::ostream& out, const InferenceRequest& request)
{
  out << "[0x" << std::addressof(request) << "] "
      << "request id: " << request.Id() << ", model: " << request.ModelName()
      << ", requested version: " << request.RequestedModelVersion()
      << ", actual version: " << request.ActualModelVersion() << ", flags: 0x"
      << std::hex << request.Flags() << std::dec
      << ", correlation id: " << request.CorrelationId()
      << ", batch size: " << request.BatchSize()
      << ", priority: " << request.Priority()
      << ", timeout (us): " << request.TimeoutMicroseconds() << std::endl;

  out << "original inputs:" << std::endl;
  for (const auto& itr : request.OriginalInputs()) {
    out << "[0x" << std::addressof(itr.second) << "] " << itr.second
        << std::endl;
  }

  out << "override inputs:" << std::endl;
  for (const auto& itr : request.OverrideInputs()) {
    out << "[0x" << itr.second.get() << "] " << *itr.second << std::endl;
  }

  out << "inputs:" << std::endl;
  for (const auto& itr : request.ImmutableInputs()) {
    out << "[0x" << itr.second << "] " << *itr.second << std::endl;
  }

  out << "original requested outputs:" << std::endl;
  for (const auto& name : request.OriginalRequestedOutputs()) {
    out << name << std::endl;
  }

  out << "requested outputs:" << std::endl;
  for (const auto& name : request.ImmutableRequestedOutputs()) {
    out << name << std::endl;
  }

  return out;
}

std::ostream&
operator<<(std::ostream& out, const InferenceRequest::Input& input)
{
  out << "input: " << input.Name()
      << ", type: " << DataTypeToProtocolString(input.DType())
      << ", original shape: " << DimsListToString(input.OriginalShape())
      << ", shape: " << DimsListToString(input.Shape());
  if (input.IsShapeTensor()) {
    out << ", is_shape_tensor: True";
  }
  return out;
}

}}  // namespace nvidia::inferenceserver
