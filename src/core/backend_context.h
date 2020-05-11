// Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

#include <list>
#include <memory>
#include <string>
#include <vector>
#include "src/core/infer_request.h"
#include "src/core/infer_response.h"
#include "src/core/memory.h"
#include "src/core/model_config.h"
#include "src/core/status.h"

#ifdef TRTIS_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRTIS_ENABLE_GPU

namespace nvidia { namespace inferenceserver {

class InferenceBackend;

//
// BackendContext
//
struct BackendContext {
 public:
#ifndef TRTIS_ENABLE_GPU
  using cudaStream_t = void*;
#endif  // !TRTIS_ENABLE_GPU

  // GPU device number that indicates that no gpu is available for a
  // context (which is an invalid state since TensorRT requires a
  // GPU).
  static constexpr int NO_GPU_DEVICE = -1;

  // Max batch size value that indicates batching is not supported.
  static constexpr int NO_BATCHING = 0;

  BackendContext(
      const std::string& name, const int gpu_device, const int max_batch_size,
      const bool enable_pinned_input, const bool enable_pinned_output);

  virtual ~BackendContext();

  // Create the CUDA stream for data transfer operations. If 'stream' is
  // nullptr, the stream will be created on 'stream_'. Have no effect if GPU
  // support is disabled.
  Status CreateCudaStream(
      const int cuda_stream_priority = 0, cudaStream_t* stream = nullptr);

  // Run model to execute one or more requests. This function assumes
  // that it is only called by the single runner thread that is
  // assigned to this context. This function takes ownership of
  // 'requests' and is responsible for generating responses and
  // releasing the requests.
  virtual void Run(
      const InferenceBackend* base,
      std::vector<std::unique_ptr<InferenceRequest>>&& requests) = 0;

  // Helper function to populate the shape value of specified shape input
  // that corresponds with the batch size. The first shape value is asssumed
  // to be the batch size. Its the user's responsibility to ensure it is called
  // only for the shape tensors.
  // Return true if cudaMemcpyAsync is called, and the caller should call
  // cudaStreamSynchronize before using the data. Otherwise, return false.
  bool SetShapeInputBuffer(
      const std::string& name, const int32_t total_batch_size,
      const int expected_byte_size, const bool support_batching,
      std::unique_ptr<InferenceRequest>& request,
      TRITONSERVER_MemoryType dst_memory_type, int64_t dst_memory_type_id,
      char* input_buffer);

  // Helper function to set output buffer for a shape tensor. It is
  // callers resposibilty to ensure this method is called only for the
  // shape tensors. Return true if cudaMemcpyAsync is called, and the
  // caller should call cudaStreamSynchronize before using the
  // data. Otherwise, return false.
  bool SetOutputShapeTensorBuffer(
      const std::string& name, const int32_t* content,
      std::vector<int64_t>& content_shape, const bool support_batching,
      TRITONSERVER_MemoryType src_memory_type, int64_t src_memory_type_id,
      std::vector<std::unique_ptr<InferenceRequest>>* requests);

  // Check if output tensor produced by a model is compatible with the
  // model configuration.  Dimensions with variable size in the model
  // configuration can support any size in the corresponding output
  // tensor dimension.
  //
  // \param supports_batching If True then the configuration expects
  // the model to support batching and so the shape must have the
  // appropriate batch dimension.
  Status CompareOutputDims(
      const std::string& tensor_name, const std::vector<int64_t>& model_shape,
      const DimsList& dims, const bool supports_batching);

  // Name of the model instance
  std::string name_;

  // The GPU index active when this context was created.
  const int gpu_device_;

  // Maximum batch size to allow. This is the minimum of what is
  // supported by the model and what is requested in the
  // configuration.
  const int max_batch_size_;

  // Whether to use indirect pinned buffer for the corresponding data copy type.
  const bool enable_pinned_input_;
  const bool enable_pinned_output_;

  // The stream where data transfer operations are executed on.
  cudaStream_t stream_;
};

//
// BackendResponder
//
class BackendResponder {
 public:
  explicit BackendResponder(
      const std::vector<std::unique_ptr<InferenceRequest>>& requests,
      std::vector<std::unique_ptr<InferenceResponse>>* responses,
      const int max_batch_size, const bool pinned_enabled, cudaStream_t stream)
      : need_sync_(false), requests_(requests), responses_(responses),
        max_batch_size_(max_batch_size), pinned_enabled_(pinned_enabled),
        stream_(stream), pending_pinned_byte_size_(0)
  {
  }

  // Process all responses for a named output tensor.
  void ProcessTensor(
      const std::string& name, const DataType datatype,
      std::vector<int64_t>& batchn_shape, const char* buffer,
      const TRITONSERVER_MemoryType memory_type, const int64_t memory_type_id);

  // Finalize processing of all responses for all output
  // tensors. Return true if cudaMemcpyAsync is called, and the caller
  // should call cudaStreamSynchronize before using the data.
  bool Finalize();

 private:
  bool FlushPendingPinned(
      const char* tensor_buffer,
      const TRITONSERVER_MemoryType tensor_memory_type,
      const int64_t tensor_memory_type_id);
  bool SetFixedSizeOutputBuffer(
      std::unique_ptr<InferenceResponse>* response,
      InferenceResponse::Output* response_output, const size_t tensor_byte_size,
      const size_t tensor_offset, const char* tensor_buffer,
      const TRITONSERVER_MemoryType tensor_memory_type,
      const int64_t tensor_memory_type_id,
      const TRITONSERVER_MemoryType use_pinned_memory_type);

  bool need_sync_;
  const std::vector<std::unique_ptr<InferenceRequest>>& requests_;
  std::vector<std::unique_ptr<InferenceResponse>>* responses_;
  const int max_batch_size_;
  const bool pinned_enabled_;
  cudaStream_t stream_;

  using ResponsesList = std::list<std::pair<
      std::unique_ptr<InferenceResponse>*, InferenceResponse::Output*>>;

  size_t pending_pinned_byte_size_;
  size_t pending_pinned_offset_;
  ResponsesList pending_pinned_outputs_;

  // Pinned memories that need to live over the lifetime of this
  // BackendResponder object.
  std::list<std::unique_ptr<AllocatedMemory>> pinned_memories_;

  // Pinned memory buffers and the corresponding response outputs
  // where the final copy to the response is deferred until Finalize()
  // after waiting for all in-flight copies.
  struct DeferredPinned {
    DeferredPinned(
        std::unique_ptr<AllocatedMemory>&& pinned_memory,
        ResponsesList&& responses)
        : pinned_memory_(std::move(pinned_memory)),
          responses_(std::move(responses))
    {
    }
    std::unique_ptr<AllocatedMemory> pinned_memory_;
    ResponsesList responses_;
  };

  std::list<DeferredPinned> deferred_pinned_;
};

//
// BackendInputCollector
//
class BackendInputCollector {
 public:
  explicit BackendInputCollector(
      const std::vector<std::unique_ptr<InferenceRequest>>& requests,
      std::vector<std::unique_ptr<InferenceResponse>>* responses,
      const bool pinned_enabled, cudaStream_t stream)
      : need_sync_(false), requests_(requests), responses_(responses),
        pinned_enabled_(pinned_enabled), stream_(stream),
        pending_pinned_byte_size_(0)
  {
  }

  // Process all requests for a named input tensor.
  void ProcessTensor(
      const std::string& name, const DataType datatype,
      const std::vector<int64_t>& batch1_shape, char* buffer,
      const size_t buffer_byte_size, const TRITONSERVER_MemoryType memory_type,
      const int64_t memory_type_id);

  // Finalize processing of all requests for all input tensors. Return
  // true if cudaMemcpyAsync is called, and the caller should call
  // cudaStreamSynchronize before using the data.
  bool Finalize();

 private:
  bool FlushPendingPinned(
      char* tensor_buffer, const size_t tensor_buffer_byte_size,
      const TRITONSERVER_MemoryType tensor_memory_type,
      const int64_t tensor_memory_type_id);
  bool SetFixedSizeInputTensor(
      const InferenceRequest::Input* request_input,
      const size_t tensor_buffer_offset, char* tensor_buffer,
      const size_t tensor_buffer_byte_size,
      const TRITONSERVER_MemoryType tensor_memory_type,
      const int64_t tensor_memory_type_id,
      const TRITONSERVER_MemoryType use_pinned_memory_type,
      std::unique_ptr<InferenceResponse>* response);

  bool need_sync_;
  const std::vector<std::unique_ptr<InferenceRequest>>& requests_;
  std::vector<std::unique_ptr<InferenceResponse>>* responses_;
  const bool pinned_enabled_;
  cudaStream_t stream_;

  using RequestsList = std::list<std::pair<
      std::unique_ptr<InferenceResponse>*, const InferenceRequest::Input*>>;

  size_t pending_pinned_byte_size_;
  size_t pending_pinned_offset_;
  RequestsList pending_pinned_inputs_;

  // Pinned memories that need to live over the lifetime of this
  // BackendResponder object.
  std::list<std::unique_ptr<AllocatedMemory>> pinned_memories_;

  // Pinned memory buffers and the corresponding request_inputs where
  // the final copy to the tensor is deferred until Finalize() after
  // waiting for all in-flight copies.
  struct DeferredPinned {
    DeferredPinned(
        std::unique_ptr<AllocatedMemory>&& pinned_memory, char* tensor_buffer,
        const size_t tensor_buffer_offset,
        const TRITONSERVER_MemoryType tensor_memory_type,
        const int64_t tensor_memory_id, RequestsList&& requests)
        : pinned_memory_(std::move(pinned_memory)),
          tensor_buffer_(tensor_buffer),
          tensor_buffer_offset_(tensor_buffer_offset),
          tensor_memory_type_(tensor_memory_type),
          tensor_memory_id_(tensor_memory_id), requests_(std::move(requests))
    {
    }
    std::unique_ptr<AllocatedMemory> pinned_memory_;
    char* tensor_buffer_;
    const size_t tensor_buffer_offset_;
    const TRITONSERVER_MemoryType tensor_memory_type_;
    const int64_t tensor_memory_id_;
    RequestsList requests_;
  };

  std::list<DeferredPinned> deferred_pinned_;
};

// This function will return a tensor's contents as a contiguous
// chunk. In some cases this will require copying the data. If that
// happens, 'contiguous_buffer' will be set to hold the contiguous
// chunk and 'cuda_copy' will be set to indicate whether CUDA copy is
// conducted.  The data copy can be avoided if the input is already in
// a contiguous chunk and the input is located in memory type and id
// specified.
Status GetContiguousInputContent(
    const InferenceRequest::Input* rinput, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id, const char** content, size_t* content_byte_size,
    std::unique_ptr<AllocatedMemory>* contiguous_buffer, cudaStream_t stream,
    bool* cuda_copy);


}}  // namespace nvidia::inferenceserver
