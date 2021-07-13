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

#include <string>
#include <vector>
#include "src/core/backend.h"
#include "src/core/cuda_utils.h"
#include "src/core/infer_parameter.h"
#include "src/core/infer_request.h"
#include "src/core/infer_response.h"
#include "src/core/infer_stats.h"
#include "src/core/logging.h"
#include "src/core/metrics.h"
#include "src/core/model_config.h"
#include "src/core/model_config_utils.h"
#include "src/core/model_repository_manager.h"
#include "src/core/nvtx.h"
#include "src/core/rate_limiter.h"
#include "src/core/response_allocator.h"
#include "src/core/server.h"
#include "src/core/server_message.h"
#include "src/core/status.h"
#include "src/core/tritonserver_apis.h"

#define TRITONJSON_STATUSTYPE nvidia::inferenceserver::Status
#define TRITONJSON_STATUSRETURN(M)        \
  return nvidia::inferenceserver::Status( \
      nvidia::inferenceserver::Status::Code::INTERNAL, (M))
#define TRITONJSON_STATUSSUCCESS nvidia::inferenceserver::Status::Success
#include "triton/common/table_printer.h"
#include "triton/common/triton_json.h"

namespace ni = nvidia::inferenceserver;

namespace {

std::string
ResourceString(const std::string& name, const int count, const int device_id)
{
  return std::string(
      "{\"name\":\"" + name + "\", \"count\":" + std::to_string(count) +
      " \"device\":" + std::to_string(device_id) + "}");
}

std::string
RateLimitModeToString(const ni::RateLimitMode rate_limit_mode)
{
  std::string rl_mode_str("<unknown>");
  switch (rate_limit_mode) {
    case ni::RateLimitMode::RL_EXEC_COUNT: {
      rl_mode_str = "EXEC_COUNT";
      break;
    }
    case ni::RateLimitMode::RL_OFF: {
      rl_mode_str = "OFF";
      break;
    }
  }
  return rl_mode_str;
}

//
// TritonServerError
//
// Implementation for TRITONSERVER_Error.
//
class TritonServerError {
 public:
  static TRITONSERVER_Error* Create(
      TRITONSERVER_Error_Code code, const char* msg);
  static TRITONSERVER_Error* Create(
      TRITONSERVER_Error_Code code, const std::string& msg);
  static TRITONSERVER_Error* Create(const ni::Status& status);

  TRITONSERVER_Error_Code Code() const { return code_; }
  const std::string& Message() const { return msg_; }

 private:
  TritonServerError(TRITONSERVER_Error_Code code, const std::string& msg)
      : code_(code), msg_(msg)
  {
  }
  TritonServerError(TRITONSERVER_Error_Code code, const char* msg)
      : code_(code), msg_(msg)
  {
  }

  TRITONSERVER_Error_Code code_;
  const std::string msg_;
};

TRITONSERVER_Error*
TritonServerError::Create(TRITONSERVER_Error_Code code, const char* msg)
{
  return reinterpret_cast<TRITONSERVER_Error*>(
      new TritonServerError(code, msg));
}

TRITONSERVER_Error*
TritonServerError::Create(TRITONSERVER_Error_Code code, const std::string& msg)
{
  return reinterpret_cast<TRITONSERVER_Error*>(
      new TritonServerError(code, msg));
}

TRITONSERVER_Error*
TritonServerError::Create(const ni::Status& status)
{
  // If 'status' is success then return nullptr as that indicates
  // success
  if (status.IsOk()) {
    return nullptr;
  }

  return Create(
      ni::StatusCodeToTritonCode(status.StatusCode()), status.Message());
}

#define RETURN_IF_STATUS_ERROR(S)                 \
  do {                                            \
    const ni::Status& status__ = (S);             \
    if (!status__.IsOk()) {                       \
      return TritonServerError::Create(status__); \
    }                                             \
  } while (false)

//
// TritonServerMetrics
//
// Implementation for TRITONSERVER_Metrics.
//
class TritonServerMetrics {
 public:
  TritonServerMetrics() = default;
  TRITONSERVER_Error* Serialize(const char** base, size_t* byte_size);

 private:
  std::string serialized_;
};

TRITONSERVER_Error*
TritonServerMetrics::Serialize(const char** base, size_t* byte_size)
{
#ifdef TRITON_ENABLE_METRICS
  serialized_ = ni::Metrics::SerializedMetrics();
  *base = serialized_.c_str();
  *byte_size = serialized_.size();
  return nullptr;  // Success
#else
  *base = nullptr;
  *byte_size = 0;
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "metrics not supported");
#endif  // TRITON_ENABLE_METRICS
}

//
// TritonServerOptions
//
// Implementation for TRITONSERVER_ServerOptions.
//
class TritonServerOptions {
 public:
  TritonServerOptions();

  const std::string& ServerId() const { return server_id_; }
  void SetServerId(const char* id) { server_id_ = id; }

  const std::set<std::string>& ModelRepositoryPaths() const
  {
    return repo_paths_;
  }
  void SetModelRepositoryPath(const char* p) { repo_paths_.insert(p); }

  ni::ModelControlMode ModelControlMode() const { return model_control_mode_; }
  void SetModelControlMode(ni::ModelControlMode m) { model_control_mode_ = m; }

  const std::set<std::string>& StartupModels() const { return models_; }
  void SetStartupModel(const char* m) { models_.insert(m); }

  bool ExitOnError() const { return exit_on_error_; }
  void SetExitOnError(bool b) { exit_on_error_ = b; }

  bool StrictModelConfig() const { return strict_model_config_; }
  void SetStrictModelConfig(bool b)
  {
    strict_model_config_ = b;
    // Note the condition is reverted due to setting name is different
    AddBackendConfig(
        std::string(), "auto-complete-config", b ? "false" : "true");
  }

  ni::RateLimitMode RateLimiterMode() const { return rate_limit_mode_; }
  void SetRateLimiterMode(ni::RateLimitMode m) { rate_limit_mode_ = m; }

  TRITONSERVER_Error* AddRateLimiterResource(
      const std::string& resource, const size_t count, const int device);

  // The resource map is the map from device id to the map of
  // of resources with their respective counts for that device.
  const ni::RateLimiter::ResourceMap& RateLimiterResources() const
  {
    return rate_limit_resource_map_;
  }

  uint64_t PinnedMemoryPoolByteSize() const { return pinned_memory_pool_size_; }
  void SetPinnedMemoryPoolByteSize(uint64_t s) { pinned_memory_pool_size_ = s; }


  const std::map<int, uint64_t>& CudaMemoryPoolByteSize() const
  {
    return cuda_memory_pool_size_;
  }
  void SetCudaMemoryPoolByteSize(int id, uint64_t s)
  {
    cuda_memory_pool_size_[id] = s;
  }

  double MinSupportedComputeCapability() const
  {
    return min_compute_capability_;
  }
  void SetMinSupportedComputeCapability(double c)
  {
    min_compute_capability_ = c;
    AddBackendConfig(
        std::string(), "min-compute-capability", std::to_string(c));
  }

  bool StrictReadiness() const { return strict_readiness_; }
  void SetStrictReadiness(bool b) { strict_readiness_ = b; }

  unsigned int ExitTimeout() const { return exit_timeout_; }
  void SetExitTimeout(unsigned int t) { exit_timeout_ = t; }

  unsigned int BufferManagerThreadCount() const
  {
    return buffer_manager_thread_count_;
  }
  void SetBufferManagerThreadCount(unsigned int c)
  {
    buffer_manager_thread_count_ = c;
  }

  bool Metrics() const { return metrics_; }
  void SetMetrics(bool b) { metrics_ = b; }

  bool GpuMetrics() const { return gpu_metrics_; }
  void SetGpuMetrics(bool b) { gpu_metrics_ = b; }

  const std::string& BackendDir() const { return backend_dir_; }
  void SetBackendDir(const std::string& bd)
  {
    backend_dir_ = bd;
    AddBackendConfig(std::string(), "backend-directory", bd);
  }

  const std::string& RepoAgentDir() const { return repoagent_dir_; }
  void SetRepoAgentDir(const std::string& rad) { repoagent_dir_ = rad; }

  // The backend config map is a map from backend name to the
  // setting=value pairs for that backend. The empty backend name ("")
  // is used to communicate configuration information that is used
  // internally.
  const ni::BackendCmdlineConfigMap& BackendCmdlineConfigMap() const
  {
    return backend_cmdline_config_map_;
  }
  TRITONSERVER_Error* AddBackendConfig(
      const std::string& backend_name, const std::string& setting,
      const std::string& value);

  TRITONSERVER_Error* SetHostPolicy(
      const std::string& policy_name, const std::string& setting,
      const std::string& value);
  const ni::HostPolicyCmdlineConfigMap& HostPolicyCmdlineConfigMap() const
  {
    return host_policy_map_;
  }

  bool TensorFlowSoftPlacement() const { return tf_soft_placement_; }
  void SetTensorFlowSoftPlacement(bool b) { tf_soft_placement_ = b; }

  float TensorFlowGpuMemoryFraction() const { return tf_gpu_mem_fraction_; }
  void SetTensorFlowGpuMemoryFraction(float f) { tf_gpu_mem_fraction_ = f; }

 private:
  std::string server_id_;
  std::set<std::string> repo_paths_;
  ni::ModelControlMode model_control_mode_;
  std::set<std::string> models_;
  bool exit_on_error_;
  bool strict_model_config_;
  bool strict_readiness_;
  ni::RateLimitMode rate_limit_mode_;
  ni::RateLimiter::ResourceMap rate_limit_resource_map_;
  bool metrics_;
  bool gpu_metrics_;
  unsigned int exit_timeout_;
  uint64_t pinned_memory_pool_size_;
  unsigned int buffer_manager_thread_count_;
  std::map<int, uint64_t> cuda_memory_pool_size_;
  double min_compute_capability_;
  std::string backend_dir_;
  std::string repoagent_dir_;
  ni::BackendCmdlineConfigMap backend_cmdline_config_map_;
  ni::HostPolicyCmdlineConfigMap host_policy_map_;

  bool tf_soft_placement_;
  float tf_gpu_mem_fraction_;
};

TritonServerOptions::TritonServerOptions()
    : server_id_("triton"),
      model_control_mode_(ni::ModelControlMode::MODE_POLL),
      exit_on_error_(true), strict_model_config_(true), strict_readiness_(true),
      rate_limit_mode_(ni::RateLimitMode::RL_EXEC_COUNT), metrics_(true),
      gpu_metrics_(true), exit_timeout_(30), pinned_memory_pool_size_(1 << 28),
      buffer_manager_thread_count_(0),
#ifdef TRITON_ENABLE_GPU
      min_compute_capability_(TRITON_MIN_COMPUTE_CAPABILITY),
#else
      min_compute_capability_(0),
#endif  // TRITON_ENABLE_GPU
      backend_dir_("/opt/tritonserver/backends"),
      repoagent_dir_("/opt/tritonserver/repoagents"), tf_soft_placement_(true),
      tf_gpu_mem_fraction_(0)
{
#ifndef TRITON_ENABLE_METRICS
  metrics_ = false;
  gpu_metrics_ = false;
#endif  // TRITON_ENABLE_METRICS

#ifndef TRITON_ENABLE_METRICS_GPU
  gpu_metrics_ = false;
#endif  // TRITON_ENABLE_METRICS_GPU
}

TRITONSERVER_Error*
ParseBoolOption(std::string arg, bool* val)
{
  std::transform(arg.begin(), arg.end(), arg.begin(), [](unsigned char c) {
    return std::tolower(c);
  });

  if ((arg == "true") || (arg == "on") || (arg == "1")) {
    *val = true;
    return nullptr;  // success
  }
  if ((arg == "false") || (arg == "off") || (arg == "0")) {
    *val = false;
    return nullptr;  // success
  }

  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_INVALID_ARG,
      std::string("invalid value for bool option: '" + arg + "'").c_str());
}

TRITONSERVER_Error*
ParseFloatOption(const std::string arg, float* val)
{
  try {
    *val = std::stof(arg);
  }
  catch (const std::invalid_argument& ia) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string("invalid value for float option: '" + arg + "'").c_str());
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
TritonServerOptions::AddRateLimiterResource(
    const std::string& name, const size_t count, const int device)
{
  auto ditr = rate_limit_resource_map_.find(device);
  if (ditr == rate_limit_resource_map_.end()) {
    ditr = rate_limit_resource_map_
               .emplace(device, std::map<std::string, size_t>())
               .first;
  }
  auto ritr = ditr->second.find(name);
  if (ritr == ditr->second.end()) {
    ditr->second.emplace(name, count).first;
  } else {
    // If already present then store the minimum of the two.
    if (ritr->second > count) {
      ritr->second = count;
    }
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
TritonServerOptions::AddBackendConfig(
    const std::string& backend_name, const std::string& setting,
    const std::string& value)
{
  ni::BackendCmdlineConfig& cc = backend_cmdline_config_map_[backend_name];
  cc.push_back(std::make_pair(setting, value));

  // FIXME this TF specific parsing and option setting and also the
  // corresponding functions in InferenceServer should be removed or
  // moved to backend once TF backend is moved to TritonBackend.
  if (backend_name == "tensorflow") {
    if (setting == "allow-soft-placement") {
      return ParseBoolOption(value, &tf_soft_placement_);
    } else if (setting == "gpu-memory-fraction") {
      return ParseFloatOption(value, &tf_gpu_mem_fraction_);
    }
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
TritonServerOptions::SetHostPolicy(
    const std::string& policy_name, const std::string& setting,
    const std::string& value)
{
  // Check if supported setting is passed
  if ((setting != "numa-node") && (setting != "cpu-cores")) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        std::string(
            "Unsupported host policy setting '" + setting +
            "' is specified, supported settings are 'numa-node', 'cpu-cores'")
            .c_str());
  }

  ni::HostPolicyCmdlineConfig& hp = host_policy_map_[policy_name];
  hp[setting] = value;

  return nullptr;  // success
}

#define SetDurationStat(DOC, PARENT, STAT_NAME, COUNT, NS)   \
  do {                                                       \
    triton::common::TritonJson::Value dstat(                 \
        DOC, triton::common::TritonJson::ValueType::OBJECT); \
    dstat.AddUInt("count", (COUNT));                         \
    dstat.AddUInt("ns", (NS));                               \
    PARENT.Add(STAT_NAME, std::move(dstat));                 \
  } while (false)

}  // namespace

#ifdef __cplusplus
extern "C" {
#endif

//
// TRITONSERVER API Version
//
TRITONSERVER_Error*
TRITONSERVER_ApiVersion(uint32_t* major, uint32_t* minor)
{
  *major = TRITONSERVER_API_VERSION_MAJOR;
  *minor = TRITONSERVER_API_VERSION_MINOR;
  return nullptr;  // success
}

//
// TRITONSERVER_DataType
//
const char*
TRITONSERVER_DataTypeString(TRITONSERVER_DataType datatype)
{
  switch (datatype) {
    case TRITONSERVER_TYPE_BOOL:
      return "BOOL";
    case TRITONSERVER_TYPE_UINT8:
      return "UINT8";
    case TRITONSERVER_TYPE_UINT16:
      return "UINT16";
    case TRITONSERVER_TYPE_UINT32:
      return "UINT32";
    case TRITONSERVER_TYPE_UINT64:
      return "UINT64";
    case TRITONSERVER_TYPE_INT8:
      return "INT8";
    case TRITONSERVER_TYPE_INT16:
      return "INT16";
    case TRITONSERVER_TYPE_INT32:
      return "INT32";
    case TRITONSERVER_TYPE_INT64:
      return "INT64";
    case TRITONSERVER_TYPE_FP16:
      return "FP16";
    case TRITONSERVER_TYPE_FP32:
      return "FP32";
    case TRITONSERVER_TYPE_FP64:
      return "FP64";
    case TRITONSERVER_TYPE_BYTES:
      return "BYTES";
    default:
      break;
  }

  return "<invalid>";
}

TRITONSERVER_DataType
TRITONSERVER_StringToDataType(const char* dtype)
{
  const size_t len = strlen(dtype);
  return ni::DataTypeToTriton(ni::ProtocolStringToDataType(dtype, len));
}

uint32_t
TRITONSERVER_DataTypeByteSize(TRITONSERVER_DataType datatype)
{
  switch (datatype) {
    case TRITONSERVER_TYPE_BOOL:
    case TRITONSERVER_TYPE_INT8:
    case TRITONSERVER_TYPE_UINT8:
      return 1;
    case TRITONSERVER_TYPE_INT16:
    case TRITONSERVER_TYPE_UINT16:
    case TRITONSERVER_TYPE_FP16:
      return 2;
    case TRITONSERVER_TYPE_INT32:
    case TRITONSERVER_TYPE_UINT32:
    case TRITONSERVER_TYPE_FP32:
      return 4;
    case TRITONSERVER_TYPE_INT64:
    case TRITONSERVER_TYPE_UINT64:
    case TRITONSERVER_TYPE_FP64:
      return 8;
    case TRITONSERVER_TYPE_BYTES:
      return 0;
    default:
      break;
  }

  return 0;
}

//
// TRITONSERVER_MemoryType
//
const char*
TRITONSERVER_MemoryTypeString(TRITONSERVER_MemoryType memtype)
{
  switch (memtype) {
    case TRITONSERVER_MEMORY_CPU:
      return "CPU";
    case TRITONSERVER_MEMORY_CPU_PINNED:
      return "CPU_PINNED";
    case TRITONSERVER_MEMORY_GPU:
      return "GPU";
    default:
      break;
  }

  return "<invalid>";
}

//
// TRITONSERVER_ParameterType
//
const char*
TRITONSERVER_ParameterTypeString(TRITONSERVER_ParameterType paramtype)
{
  switch (paramtype) {
    case TRITONSERVER_PARAMETER_STRING:
      return "STRING";
    case TRITONSERVER_PARAMETER_INT:
      return "INT";
    case TRITONSERVER_PARAMETER_BOOL:
      return "BOOL";
    default:
      break;
  }

  return "<invalid>";
}

//
// TRITONSERVER_InstanceGroupKind
//
const char*
TRITONSERVER_InstanceGroupKindString(TRITONSERVER_InstanceGroupKind kind)
{
  switch (kind) {
    case TRITONSERVER_INSTANCEGROUPKIND_AUTO:
      return "AUTO";
    case TRITONSERVER_INSTANCEGROUPKIND_CPU:
      return "CPU";
    case TRITONSERVER_INSTANCEGROUPKIND_GPU:
      return "GPU";
    case TRITONSERVER_INSTANCEGROUPKIND_MODEL:
      return "MODEL";
    default:
      break;
  }

  return "<invalid>";
}

//
// TRITONSERVER_Log
//
bool
TRITONSERVER_LogIsEnabled(TRITONSERVER_LogLevel level)
{
  switch (level) {
    case TRITONSERVER_LOG_INFO:
      return LOG_INFO_IS_ON;
    case TRITONSERVER_LOG_WARN:
      return LOG_WARNING_IS_ON;
    case TRITONSERVER_LOG_ERROR:
      return LOG_ERROR_IS_ON;
    case TRITONSERVER_LOG_VERBOSE:
      return LOG_VERBOSE_IS_ON(1);
  }

  return false;
}

TRITONSERVER_Error*
TRITONSERVER_LogMessage(
    TRITONSERVER_LogLevel level, const char* filename, const int line,
    const char* msg)
{
  switch (level) {
    case TRITONSERVER_LOG_INFO:
      LOG_INFO_FL(filename, line) << msg;
      return nullptr;
    case TRITONSERVER_LOG_WARN:
      LOG_WARNING_FL(filename, line) << msg;
      return nullptr;
    case TRITONSERVER_LOG_ERROR:
      LOG_ERROR_FL(filename, line) << msg;
      return nullptr;
    case TRITONSERVER_LOG_VERBOSE:
      LOG_VERBOSE_FL(1, filename, line) << msg;
      return nullptr;
    default:
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          std::string("unknown logging level '" + std::to_string(level) + "'")
              .c_str());
  }
}

//
// TRITONSERVER_Error
//
TRITONSERVER_Error*
TRITONSERVER_ErrorNew(TRITONSERVER_Error_Code code, const char* msg)
{
  return reinterpret_cast<TRITONSERVER_Error*>(
      TritonServerError::Create(code, msg));
}

void
TRITONSERVER_ErrorDelete(TRITONSERVER_Error* error)
{
  TritonServerError* lerror = reinterpret_cast<TritonServerError*>(error);
  delete lerror;
}

TRITONSERVER_Error_Code
TRITONSERVER_ErrorCode(TRITONSERVER_Error* error)
{
  TritonServerError* lerror = reinterpret_cast<TritonServerError*>(error);
  return lerror->Code();
}

const char*
TRITONSERVER_ErrorCodeString(TRITONSERVER_Error* error)
{
  TritonServerError* lerror = reinterpret_cast<TritonServerError*>(error);
  return ni::Status::CodeString(ni::TritonCodeToStatusCode(lerror->Code()));
}

const char*
TRITONSERVER_ErrorMessage(TRITONSERVER_Error* error)
{
  TritonServerError* lerror = reinterpret_cast<TritonServerError*>(error);
  return lerror->Message().c_str();
}

//
// TRITONSERVER_ResponseAllocator
//
TRITONSERVER_Error*
TRITONSERVER_ResponseAllocatorNew(
    TRITONSERVER_ResponseAllocator** allocator,
    TRITONSERVER_ResponseAllocatorAllocFn_t alloc_fn,
    TRITONSERVER_ResponseAllocatorReleaseFn_t release_fn,
    TRITONSERVER_ResponseAllocatorStartFn_t start_fn)
{
  *allocator = reinterpret_cast<TRITONSERVER_ResponseAllocator*>(
      new ni::ResponseAllocator(alloc_fn, release_fn, start_fn));
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ResponseAllocatorDelete(TRITONSERVER_ResponseAllocator* allocator)
{
  ni::ResponseAllocator* lalloc =
      reinterpret_cast<ni::ResponseAllocator*>(allocator);
  delete lalloc;
  return nullptr;  // Success
}

//
// TRITONSERVER_Message
//
TRITONSERVER_Error*
TRITONSERVER_MessageNewFromSerializedJson(
    TRITONSERVER_Message** message, const char* base, size_t byte_size)
{
  *message = reinterpret_cast<TRITONSERVER_Message*>(
      new ni::TritonServerMessage({base, byte_size}));
  return nullptr;
}

TRITONSERVER_Error*
TRITONSERVER_MessageDelete(TRITONSERVER_Message* message)
{
  ni::TritonServerMessage* lmessage =
      reinterpret_cast<ni::TritonServerMessage*>(message);
  delete lmessage;
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_MessageSerializeToJson(
    TRITONSERVER_Message* message, const char** base, size_t* byte_size)
{
  ni::TritonServerMessage* lmessage =
      reinterpret_cast<ni::TritonServerMessage*>(message);
  lmessage->Serialize(base, byte_size);
  return nullptr;  // Success
}

//
// TRITONSERVER_Metrics
//
TRITONSERVER_Error*
TRITONSERVER_MetricsDelete(TRITONSERVER_Metrics* metrics)
{
  TritonServerMetrics* lmetrics =
      reinterpret_cast<TritonServerMetrics*>(metrics);
  delete lmetrics;
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_MetricsFormatted(
    TRITONSERVER_Metrics* metrics, TRITONSERVER_MetricFormat format,
    const char** base, size_t* byte_size)
{
  TritonServerMetrics* lmetrics =
      reinterpret_cast<TritonServerMetrics*>(metrics);

  switch (format) {
    case TRITONSERVER_METRIC_PROMETHEUS: {
      return lmetrics->Serialize(base, byte_size);
    }

    default:
      break;
  }

  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_INVALID_ARG,
      std::string("unknown metrics format '" + std::to_string(format) + "'")
          .c_str());
}

//
// TRITONSERVER_InferenceTrace
//
const char*
TRITONSERVER_InferenceTraceLevelString(TRITONSERVER_InferenceTraceLevel level)
{
  switch (level) {
    case TRITONSERVER_TRACE_LEVEL_DISABLED:
      return "DISABLED";
    case TRITONSERVER_TRACE_LEVEL_MIN:
      return "MIN";
    case TRITONSERVER_TRACE_LEVEL_MAX:
      return "MAX";
  }

  return "<unknown>";
}

const char*
TRITONSERVER_InferenceTraceActivityString(
    TRITONSERVER_InferenceTraceActivity activity)
{
  switch (activity) {
    case TRITONSERVER_TRACE_REQUEST_START:
      return "REQUEST_START";
    case TRITONSERVER_TRACE_QUEUE_START:
      return "QUEUE_START";
    case TRITONSERVER_TRACE_COMPUTE_START:
      return "COMPUTE_START";
    case TRITONSERVER_TRACE_COMPUTE_INPUT_END:
      return "COMPUTE_INPUT_END";
    case TRITONSERVER_TRACE_COMPUTE_OUTPUT_START:
      return "COMPUTE_OUTPUT_START";
    case TRITONSERVER_TRACE_COMPUTE_END:
      return "COMPUTE_END";
    case TRITONSERVER_TRACE_REQUEST_END:
      return "REQUEST_END";
  }

  return "<unknown>";
}

TRITONSERVER_Error*
TRITONSERVER_InferenceTraceNew(
    TRITONSERVER_InferenceTrace** trace, TRITONSERVER_InferenceTraceLevel level,
    uint64_t parent_id, TRITONSERVER_InferenceTraceActivityFn_t activity_fn,
    TRITONSERVER_InferenceTraceReleaseFn_t release_fn, void* trace_userp)
{
#ifdef TRITON_ENABLE_TRACING
  ni::InferenceTrace* ltrace = new ni::InferenceTrace(
      level, parent_id, activity_fn, release_fn, trace_userp);
  *trace = reinterpret_cast<TRITONSERVER_InferenceTrace*>(ltrace);
  return nullptr;  // Success
#else
  *trace = nullptr;
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "inference tracing not supported");
#endif  // TRITON_ENABLE_TRACING
}

TRITONSERVER_Error*
TRITONSERVER_InferenceTraceDelete(TRITONSERVER_InferenceTrace* trace)
{
#ifdef TRITON_ENABLE_TRACING
  ni::InferenceTrace* ltrace = reinterpret_cast<ni::InferenceTrace*>(trace);
  delete ltrace;
  return nullptr;  // Success
#else
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "inference tracing not supported");
#endif  // TRITON_ENABLE_TRACING
}

TRITONSERVER_Error*
TRITONSERVER_InferenceTraceId(TRITONSERVER_InferenceTrace* trace, uint64_t* id)
{
#ifdef TRITON_ENABLE_TRACING
  ni::InferenceTrace* ltrace = reinterpret_cast<ni::InferenceTrace*>(trace);
  *id = ltrace->Id();
  return nullptr;  // Success
#else
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "inference tracing not supported");
#endif  // TRITON_ENABLE_TRACING
}

TRITONSERVER_Error*
TRITONSERVER_InferenceTraceParentId(
    TRITONSERVER_InferenceTrace* trace, uint64_t* parent_id)
{
#ifdef TRITON_ENABLE_TRACING
  ni::InferenceTrace* ltrace = reinterpret_cast<ni::InferenceTrace*>(trace);
  *parent_id = ltrace->ParentId();
  return nullptr;  // Success
#else
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "inference tracing not supported");
#endif  // TRITON_ENABLE_TRACING
}

TRITONSERVER_Error*
TRITONSERVER_InferenceTraceModelName(
    TRITONSERVER_InferenceTrace* trace, const char** model_name)
{
#ifdef TRITON_ENABLE_TRACING
  ni::InferenceTrace* ltrace = reinterpret_cast<ni::InferenceTrace*>(trace);
  *model_name = ltrace->ModelName().c_str();
  return nullptr;  // Success
#else
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "inference tracing not supported");
#endif  // TRITON_ENABLE_TRACING
}

TRITONSERVER_Error*
TRITONSERVER_InferenceTraceModelVersion(
    TRITONSERVER_InferenceTrace* trace, int64_t* model_version)
{
#ifdef TRITON_ENABLE_TRACING
  ni::InferenceTrace* ltrace = reinterpret_cast<ni::InferenceTrace*>(trace);
  *model_version = ltrace->ModelVersion();
  return nullptr;  // Success
#else
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "inference tracing not supported");
#endif  // TRITON_ENABLE_TRACING
}

//
// TRITONSERVER_ServerOptions
//
TRITONSERVER_Error*
TRITONSERVER_ServerOptionsNew(TRITONSERVER_ServerOptions** options)
{
  *options =
      reinterpret_cast<TRITONSERVER_ServerOptions*>(new TritonServerOptions());
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ServerOptionsDelete(TRITONSERVER_ServerOptions* options)
{
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  delete loptions;
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetServerId(
    TRITONSERVER_ServerOptions* options, const char* server_id)
{
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  loptions->SetServerId(server_id);
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetModelRepositoryPath(
    TRITONSERVER_ServerOptions* options, const char* model_repository_path)
{
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  loptions->SetModelRepositoryPath(model_repository_path);
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetModelControlMode(
    TRITONSERVER_ServerOptions* options, TRITONSERVER_ModelControlMode mode)
{
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);

  // convert mode from TRITONSERVER_ to nvidia::inferenceserver
  switch (mode) {
    case TRITONSERVER_MODEL_CONTROL_NONE: {
      loptions->SetModelControlMode(ni::ModelControlMode::MODE_NONE);
      break;
    }
    case TRITONSERVER_MODEL_CONTROL_POLL: {
      loptions->SetModelControlMode(ni::ModelControlMode::MODE_POLL);
      break;
    }
    case TRITONSERVER_MODEL_CONTROL_EXPLICIT: {
      loptions->SetModelControlMode(ni::ModelControlMode::MODE_EXPLICIT);
      break;
    }
    default: {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          std::string("unknown control mode '" + std::to_string(mode) + "'")
              .c_str());
    }
  }

  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetStartupModel(
    TRITONSERVER_ServerOptions* options, const char* model_name)
{
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  loptions->SetStartupModel(model_name);
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetExitOnError(
    TRITONSERVER_ServerOptions* options, bool exit)
{
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  loptions->SetExitOnError(exit);
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetStrictModelConfig(
    TRITONSERVER_ServerOptions* options, bool strict)
{
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  loptions->SetStrictModelConfig(strict);
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetRateLimiterMode(
    TRITONSERVER_ServerOptions* options, TRITONSERVER_RateLimitMode mode)
{
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);

  // convert mode from TRITONSERVER_ to nvidia::inferenceserver
  switch (mode) {
    case TRITONSERVER_RATE_LIMIT_EXEC_COUNT: {
      loptions->SetRateLimiterMode(ni::RateLimitMode::RL_EXEC_COUNT);
      break;
    }
    case TRITONSERVER_RATE_LIMIT_OFF: {
      loptions->SetRateLimiterMode(ni::RateLimitMode::RL_OFF);
      break;
    }
    default: {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          std::string("unknown rate limit mode '" + std::to_string(mode) + "'")
              .c_str());
    }
  }

  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ServerOptionsAddRateLimiterResource(
    TRITONSERVER_ServerOptions* options, const char* name, const size_t count,
    const int device)
{
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  return loptions->AddRateLimiterResource(name, count, device);
}


TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetPinnedMemoryPoolByteSize(
    TRITONSERVER_ServerOptions* options, uint64_t size)
{
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  loptions->SetPinnedMemoryPoolByteSize(size);
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetCudaMemoryPoolByteSize(
    TRITONSERVER_ServerOptions* options, int gpu_device, uint64_t size)
{
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  loptions->SetCudaMemoryPoolByteSize(gpu_device, size);
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetMinSupportedComputeCapability(
    TRITONSERVER_ServerOptions* options, double cc)
{
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  loptions->SetMinSupportedComputeCapability(cc);
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetStrictReadiness(
    TRITONSERVER_ServerOptions* options, bool strict)
{
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  loptions->SetStrictReadiness(strict);
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetExitTimeout(
    TRITONSERVER_ServerOptions* options, unsigned int timeout)
{
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  loptions->SetExitTimeout(timeout);
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetBufferManagerThreadCount(
    TRITONSERVER_ServerOptions* options, unsigned int thread_count)
{
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  loptions->SetBufferManagerThreadCount(thread_count);
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetLogInfo(
    TRITONSERVER_ServerOptions* options, bool log)
{
#ifdef TRITON_ENABLE_LOGGING
  // Logging is global for now...
  LOG_ENABLE_INFO(log);
  return nullptr;  // Success
#else
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "logging not supported");
#endif  // TRITON_ENABLE_LOGGING
}

// Enable or disable warning level logging.
TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetLogWarn(
    TRITONSERVER_ServerOptions* options, bool log)
{
#ifdef TRITON_ENABLE_LOGGING
  // Logging is global for now...
  LOG_ENABLE_WARNING(log);
  return nullptr;  // Success
#else
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "logging not supported");
#endif  // TRITON_ENABLE_LOGGING
}

// Enable or disable error level logging.
TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetLogError(
    TRITONSERVER_ServerOptions* options, bool log)
{
#ifdef TRITON_ENABLE_LOGGING
  // Logging is global for now...
  LOG_ENABLE_ERROR(log);
  return nullptr;  // Success
#else
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "logging not supported");
#endif  // TRITON_ENABLE_LOGGING
}

// Set verbose logging level. Level zero disables verbose logging.
TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetLogVerbose(
    TRITONSERVER_ServerOptions* options, int level)
{
#ifdef TRITON_ENABLE_LOGGING
  // Logging is global for now...
  LOG_SET_VERBOSE(level);
#else
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "logging not supported");
#endif             // TRITON_ENABLE_LOGGING
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetMetrics(
    TRITONSERVER_ServerOptions* options, bool metrics)
{
#ifdef TRITON_ENABLE_METRICS
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  loptions->SetMetrics(metrics);
  return nullptr;  // Success
#else
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "metrics not supported");
#endif  // TRITON_ENABLE_METRICS
}

TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetGpuMetrics(
    TRITONSERVER_ServerOptions* options, bool gpu_metrics)
{
#ifdef TRITON_ENABLE_METRICS
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  loptions->SetGpuMetrics(gpu_metrics);
  return nullptr;  // Success
#else
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "metrics not supported");
#endif  // TRITON_ENABLE_METRICS
}

TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetBackendDirectory(
    TRITONSERVER_ServerOptions* options, const char* backend_dir)
{
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  loptions->SetBackendDir(backend_dir);
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetRepoAgentDirectory(
    TRITONSERVER_ServerOptions* options, const char* repoagent_dir)
{
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  loptions->SetRepoAgentDir(repoagent_dir);
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetBackendConfig(
    TRITONSERVER_ServerOptions* options, const char* backend_name,
    const char* setting, const char* value)
{
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  return loptions->AddBackendConfig(backend_name, setting, value);
}

TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetHostPolicy(
    TRITONSERVER_ServerOptions* options, const char* policy_name,
    const char* setting, const char* value)
{
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);
  return loptions->SetHostPolicy(policy_name, setting, value);
}

//
// TRITONSERVER_InferenceRequest
//
TRITONSERVER_Error*
TRITONSERVER_InferenceRequestNew(
    TRITONSERVER_InferenceRequest** inference_request,
    TRITONSERVER_Server* server, const char* model_name,
    const int64_t model_version)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);

  std::shared_ptr<ni::InferenceBackend> backend;
  RETURN_IF_STATUS_ERROR(
      lserver->GetInferenceBackend(model_name, model_version, &backend));

  *inference_request = reinterpret_cast<TRITONSERVER_InferenceRequest*>(
      new ni::InferenceRequest(backend, model_version));

  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_InferenceRequestDelete(
    TRITONSERVER_InferenceRequest* inference_request)
{
  ni::InferenceRequest* lrequest =
      reinterpret_cast<ni::InferenceRequest*>(inference_request);
  delete lrequest;
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_InferenceRequestId(
    TRITONSERVER_InferenceRequest* inference_request, const char** id)
{
  ni::InferenceRequest* lrequest =
      reinterpret_cast<ni::InferenceRequest*>(inference_request);
  *id = lrequest->Id().c_str();
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_InferenceRequestSetId(
    TRITONSERVER_InferenceRequest* inference_request, const char* id)
{
  ni::InferenceRequest* lrequest =
      reinterpret_cast<ni::InferenceRequest*>(inference_request);
  lrequest->SetId(id);
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_InferenceRequestFlags(
    TRITONSERVER_InferenceRequest* inference_request, uint32_t* flags)
{
  ni::InferenceRequest* lrequest =
      reinterpret_cast<ni::InferenceRequest*>(inference_request);
  *flags = lrequest->Flags();
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_InferenceRequestSetFlags(
    TRITONSERVER_InferenceRequest* inference_request, uint32_t flags)
{
  ni::InferenceRequest* lrequest =
      reinterpret_cast<ni::InferenceRequest*>(inference_request);
  lrequest->SetFlags(flags);
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_InferenceRequestCorrelationId(
    TRITONSERVER_InferenceRequest* inference_request, uint64_t* correlation_id)
{
  ni::InferenceRequest* lrequest =
      reinterpret_cast<ni::InferenceRequest*>(inference_request);
  *correlation_id = lrequest->CorrelationId();
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_InferenceRequestSetCorrelationId(
    TRITONSERVER_InferenceRequest* inference_request, uint64_t correlation_id)
{
  ni::InferenceRequest* lrequest =
      reinterpret_cast<ni::InferenceRequest*>(inference_request);
  lrequest->SetCorrelationId(correlation_id);
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_InferenceRequestPriority(
    TRITONSERVER_InferenceRequest* inference_request, uint32_t* priority)
{
  ni::InferenceRequest* lrequest =
      reinterpret_cast<ni::InferenceRequest*>(inference_request);
  *priority = lrequest->Priority();
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_InferenceRequestSetPriority(
    TRITONSERVER_InferenceRequest* inference_request, uint32_t priority)
{
  ni::InferenceRequest* lrequest =
      reinterpret_cast<ni::InferenceRequest*>(inference_request);
  lrequest->SetPriority(priority);
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_InferenceRequestTimeoutMicroseconds(
    TRITONSERVER_InferenceRequest* inference_request, uint64_t* timeout_us)
{
  ni::InferenceRequest* lrequest =
      reinterpret_cast<ni::InferenceRequest*>(inference_request);
  *timeout_us = lrequest->TimeoutMicroseconds();
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_InferenceRequestSetTimeoutMicroseconds(
    TRITONSERVER_InferenceRequest* inference_request, uint64_t timeout_us)
{
  ni::InferenceRequest* lrequest =
      reinterpret_cast<ni::InferenceRequest*>(inference_request);
  lrequest->SetTimeoutMicroseconds(timeout_us);
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_InferenceRequestAddInput(
    TRITONSERVER_InferenceRequest* inference_request, const char* name,
    const TRITONSERVER_DataType datatype, const int64_t* shape,
    uint64_t dim_count)
{
  ni::InferenceRequest* lrequest =
      reinterpret_cast<ni::InferenceRequest*>(inference_request);
  RETURN_IF_STATUS_ERROR(lrequest->AddOriginalInput(
      name, ni::TritonToDataType(datatype), shape, dim_count));
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_InferenceRequestRemoveInput(
    TRITONSERVER_InferenceRequest* inference_request, const char* name)
{
  ni::InferenceRequest* lrequest =
      reinterpret_cast<ni::InferenceRequest*>(inference_request);
  RETURN_IF_STATUS_ERROR(lrequest->RemoveOriginalInput(name));
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_InferenceRequestRemoveAllInputs(
    TRITONSERVER_InferenceRequest* inference_request)
{
  ni::InferenceRequest* lrequest =
      reinterpret_cast<ni::InferenceRequest*>(inference_request);
  RETURN_IF_STATUS_ERROR(lrequest->RemoveAllOriginalInputs());
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_InferenceRequestAppendInputData(
    TRITONSERVER_InferenceRequest* inference_request, const char* name,
    const void* base, size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id)
{
  ni::InferenceRequest* lrequest =
      reinterpret_cast<ni::InferenceRequest*>(inference_request);

  ni::InferenceRequest::Input* input;
  RETURN_IF_STATUS_ERROR(lrequest->MutableOriginalInput(name, &input));
  RETURN_IF_STATUS_ERROR(
      input->AppendData(base, byte_size, memory_type, memory_type_id));

  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_InferenceRequestAppendInputDataWithHostPolicy(
    TRITONSERVER_InferenceRequest* inference_request, const char* name,
    const void* base, size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id, const char* host_policy_name)
{
  ni::InferenceRequest* lrequest =
      reinterpret_cast<ni::InferenceRequest*>(inference_request);

  ni::InferenceRequest::Input* input;
  RETURN_IF_STATUS_ERROR(lrequest->MutableOriginalInput(name, &input));
  RETURN_IF_STATUS_ERROR(input->AppendDataWithHostPolicy(
      base, byte_size, memory_type, memory_type_id, host_policy_name));

  return nullptr;  // Success
}


TRITONSERVER_Error*
TRITONSERVER_InferenceRequestRemoveAllInputData(
    TRITONSERVER_InferenceRequest* inference_request, const char* name)
{
  ni::InferenceRequest* lrequest =
      reinterpret_cast<ni::InferenceRequest*>(inference_request);

  ni::InferenceRequest::Input* input;
  RETURN_IF_STATUS_ERROR(lrequest->MutableOriginalInput(name, &input));
  RETURN_IF_STATUS_ERROR(input->RemoveAllData());

  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_InferenceRequestAddRequestedOutput(
    TRITONSERVER_InferenceRequest* inference_request, const char* name)
{
  ni::InferenceRequest* lrequest =
      reinterpret_cast<ni::InferenceRequest*>(inference_request);
  RETURN_IF_STATUS_ERROR(lrequest->AddOriginalRequestedOutput(name));
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_InferenceRequestRemoveRequestedOutput(
    TRITONSERVER_InferenceRequest* inference_request, const char* name)
{
  ni::InferenceRequest* lrequest =
      reinterpret_cast<ni::InferenceRequest*>(inference_request);
  RETURN_IF_STATUS_ERROR(lrequest->RemoveOriginalRequestedOutput(name));
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_InferenceRequestRemoveAllRequestedOutputs(
    TRITONSERVER_InferenceRequest* inference_request)
{
  ni::InferenceRequest* lrequest =
      reinterpret_cast<ni::InferenceRequest*>(inference_request);
  RETURN_IF_STATUS_ERROR(lrequest->RemoveAllOriginalRequestedOutputs());
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_InferenceRequestSetReleaseCallback(
    TRITONSERVER_InferenceRequest* inference_request,
    TRITONSERVER_InferenceRequestReleaseFn_t request_release_fn,
    void* request_release_userp)
{
  ni::InferenceRequest* lrequest =
      reinterpret_cast<ni::InferenceRequest*>(inference_request);
  RETURN_IF_STATUS_ERROR(
      lrequest->SetReleaseCallback(request_release_fn, request_release_userp));
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_InferenceRequestSetResponseCallback(
    TRITONSERVER_InferenceRequest* inference_request,
    TRITONSERVER_ResponseAllocator* response_allocator,
    void* response_allocator_userp,
    TRITONSERVER_InferenceResponseCompleteFn_t response_fn,
    void* response_userp)
{
  ni::InferenceRequest* lrequest =
      reinterpret_cast<ni::InferenceRequest*>(inference_request);
  ni::ResponseAllocator* lallocator =
      reinterpret_cast<ni::ResponseAllocator*>(response_allocator);
  RETURN_IF_STATUS_ERROR(lrequest->SetResponseCallback(
      lallocator, response_allocator_userp, response_fn, response_userp));
  return nullptr;  // Success
}

//
// TRITONSERVER_InferenceResponse
//
TRITONSERVER_Error*
TRITONSERVER_InferenceResponseDelete(
    TRITONSERVER_InferenceResponse* inference_response)
{
  ni::InferenceResponse* lresponse =
      reinterpret_cast<ni::InferenceResponse*>(inference_response);
  delete lresponse;
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_InferenceResponseError(
    TRITONSERVER_InferenceResponse* inference_response)
{
  ni::InferenceResponse* lresponse =
      reinterpret_cast<ni::InferenceResponse*>(inference_response);
  RETURN_IF_STATUS_ERROR(lresponse->ResponseStatus());
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_InferenceResponseModel(
    TRITONSERVER_InferenceResponse* inference_response, const char** model_name,
    int64_t* model_version)
{
  ni::InferenceResponse* lresponse =
      reinterpret_cast<ni::InferenceResponse*>(inference_response);

  *model_name = lresponse->ModelName().c_str();
  *model_version = lresponse->ActualModelVersion();

  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_InferenceResponseId(
    TRITONSERVER_InferenceResponse* inference_response, const char** request_id)
{
  ni::InferenceResponse* lresponse =
      reinterpret_cast<ni::InferenceResponse*>(inference_response);

  *request_id = lresponse->Id().c_str();

  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_InferenceResponseParameterCount(
    TRITONSERVER_InferenceResponse* inference_response, uint32_t* count)
{
  ni::InferenceResponse* lresponse =
      reinterpret_cast<ni::InferenceResponse*>(inference_response);

  const auto& parameters = lresponse->Parameters();
  *count = parameters.size();

  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_InferenceResponseParameter(
    TRITONSERVER_InferenceResponse* inference_response, const uint32_t index,
    const char** name, TRITONSERVER_ParameterType* type, const void** vvalue)
{
  ni::InferenceResponse* lresponse =
      reinterpret_cast<ni::InferenceResponse*>(inference_response);

  const auto& parameters = lresponse->Parameters();
  if (index >= parameters.size()) {
    return TritonServerError::Create(
        TRITONSERVER_ERROR_INVALID_ARG,
        "out of bounds index " + std::to_string(index) +
            std::string(": response has ") + std::to_string(parameters.size()) +
            " parameters");
  }

  const ni::InferenceParameter& param = parameters[index];

  *name = param.Name().c_str();
  *type = param.Type();
  *vvalue = param.ValuePointer();

  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_InferenceResponseOutputCount(
    TRITONSERVER_InferenceResponse* inference_response, uint32_t* count)
{
  ni::InferenceResponse* lresponse =
      reinterpret_cast<ni::InferenceResponse*>(inference_response);

  const auto& outputs = lresponse->Outputs();
  *count = outputs.size();

  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_InferenceResponseOutput(
    TRITONSERVER_InferenceResponse* inference_response, const uint32_t index,
    const char** name, TRITONSERVER_DataType* datatype, const int64_t** shape,
    uint64_t* dim_count, const void** base, size_t* byte_size,
    TRITONSERVER_MemoryType* memory_type, int64_t* memory_type_id, void** userp)
{
  ni::InferenceResponse* lresponse =
      reinterpret_cast<ni::InferenceResponse*>(inference_response);

  const auto& outputs = lresponse->Outputs();
  if (index >= outputs.size()) {
    return TritonServerError::Create(
        TRITONSERVER_ERROR_INVALID_ARG,
        "out of bounds index " + std::to_string(index) +
            std::string(": response has ") + std::to_string(outputs.size()) +
            " outputs");
  }

  const ni::InferenceResponse::Output& output = outputs[index];

  *name = output.Name().c_str();
  *datatype = ni::DataTypeToTriton(output.DType());

  const std::vector<int64_t>& oshape = output.Shape();
  *shape = &oshape[0];
  *dim_count = oshape.size();

  RETURN_IF_STATUS_ERROR(
      output.DataBuffer(base, byte_size, memory_type, memory_type_id, userp));

  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_InferenceResponseOutputClassificationLabel(
    TRITONSERVER_InferenceResponse* inference_response, const uint32_t index,
    const size_t class_index, const char** label)
{
  ni::InferenceResponse* lresponse =
      reinterpret_cast<ni::InferenceResponse*>(inference_response);

  const auto& outputs = lresponse->Outputs();
  if (index >= outputs.size()) {
    return TritonServerError::Create(
        TRITONSERVER_ERROR_INVALID_ARG,
        "out of bounds index " + std::to_string(index) +
            std::string(": response has ") + std::to_string(outputs.size()) +
            " outputs");
  }

  const ni::InferenceResponse::Output& output = outputs[index];
  RETURN_IF_STATUS_ERROR(
      lresponse->ClassificationLabel(output, class_index, label));

  return nullptr;  // Success
}

//
// TRITONSERVER_Server
//
TRITONSERVER_Error*
TRITONSERVER_ServerNew(
    TRITONSERVER_Server** server, TRITONSERVER_ServerOptions* options)
{
  ni::InferenceServer* lserver = new ni::InferenceServer();
  TritonServerOptions* loptions =
      reinterpret_cast<TritonServerOptions*>(options);

  NVTX_INITIALIZE;

#ifdef TRITON_ENABLE_METRICS
  if (loptions->Metrics()) {
    ni::Metrics::EnableMetrics();
  }
#ifdef TRITON_ENABLE_METRICS_GPU
  if (loptions->Metrics() && loptions->GpuMetrics()) {
    ni::Metrics::EnableGPUMetrics();
  }
#endif  // TRITON_ENABLE_METRICS_GPU
#endif  // TRITON_ENABLE_METRICS

  lserver->SetId(loptions->ServerId());
  lserver->SetModelRepositoryPaths(loptions->ModelRepositoryPaths());
  lserver->SetModelControlMode(loptions->ModelControlMode());
  lserver->SetStartupModels(loptions->StartupModels());
  lserver->SetStrictModelConfigEnabled(loptions->StrictModelConfig());
  lserver->SetRateLimiterMode(loptions->RateLimiterMode());
  lserver->SetRateLimiterResources(loptions->RateLimiterResources());
  lserver->SetPinnedMemoryPoolByteSize(loptions->PinnedMemoryPoolByteSize());
  lserver->SetCudaMemoryPoolByteSize(loptions->CudaMemoryPoolByteSize());
  lserver->SetMinSupportedComputeCapability(
      loptions->MinSupportedComputeCapability());
  lserver->SetStrictReadinessEnabled(loptions->StrictReadiness());
  lserver->SetExitTimeoutSeconds(loptions->ExitTimeout());
  lserver->SetBackendCmdlineConfig(loptions->BackendCmdlineConfigMap());
  lserver->SetHostPolicyCmdlineConfig(loptions->HostPolicyCmdlineConfigMap());
  lserver->SetRepoAgentDir(loptions->RepoAgentDir());
  lserver->SetBufferManagerThreadCount(loptions->BufferManagerThreadCount());

  // FIXME these should be removed once all backends use
  // BackendConfig.
  lserver->SetTensorFlowSoftPlacementEnabled(
      loptions->TensorFlowSoftPlacement());
  lserver->SetTensorFlowGPUMemoryFraction(
      loptions->TensorFlowGpuMemoryFraction());

  ni::Status status = lserver->Init();
  std::vector<std::string> options_headers;
  options_headers.emplace_back("Option");
  options_headers.emplace_back("Value");

  triton::common::TablePrinter options_table(options_headers);
  options_table.InsertRow(std::vector<std::string>{"server_id", lserver->Id()});
  options_table.InsertRow(
      std::vector<std::string>{"server_version", lserver->Version()});

  auto extensions = lserver->Extensions();
  std::string exts;
  for (const auto& ext : extensions) {
    exts.append(ext);
    exts.append(" ");
  }

  // Remove the trailing space
  if (exts.size() > 0)
    exts.pop_back();

  options_table.InsertRow(std::vector<std::string>{"server_extensions", exts});

  size_t i = 0;
  for (const auto& model_repository_path : lserver->ModelRepositoryPaths()) {
    options_table.InsertRow(std::vector<std::string>{
        "model_repository_path[" + std::to_string(i) + "]",
        model_repository_path});
    ++i;
  }

  std::string model_control_mode;
  auto control_mode = lserver->GetModelControlMode();
  switch (control_mode) {
    case ni::ModelControlMode::MODE_NONE: {
      model_control_mode = "MODE_NONE";
      break;
    }
    case ni::ModelControlMode::MODE_POLL: {
      model_control_mode = "MODE_POLL";
      break;
    }
    case ni::ModelControlMode::MODE_EXPLICIT: {
      model_control_mode = "MODE_EXPLICIT";
      break;
    }
    default: {
      model_control_mode = "<unknown>";
    }
  }
  options_table.InsertRow(
      std::vector<std::string>{"model_control_mode", model_control_mode});

  i = 0;
  for (const auto& startup_model : lserver->StartupModels()) {
    options_table.InsertRow(std::vector<std::string>{
        "startup_models_" + std::to_string(i), startup_model});
    ++i;
  }
  options_table.InsertRow(std::vector<std::string>{
      "strict_model_config",
      std::to_string(lserver->StrictModelConfigEnabled())});
  std::string rate_limit = RateLimitModeToString(lserver->RateLimiterMode());
  options_table.InsertRow(std::vector<std::string>{"rate_limit", rate_limit});
  i = 0;
  for (const auto& device_resources : lserver->RateLimiterResources()) {
    for (const auto& resource : device_resources.second) {
      options_table.InsertRow(std::vector<std::string>{
          "rate_limit_resource[" + std::to_string(i) + "]",
          ResourceString(
              resource.first, resource.second, device_resources.first)});
      ++i;
    }
  }
  options_table.InsertRow(std::vector<std::string>{
      "pinned_memory_pool_byte_size",
      std::to_string(lserver->PinnedMemoryPoolByteSize())});
  for (const auto& cuda_memory_pool : lserver->CudaMemoryPoolByteSize()) {
    options_table.InsertRow(std::vector<std::string>{
        "cuda_memory_pool_byte_size{" + std::to_string(cuda_memory_pool.first) +
            "}",
        std::to_string(cuda_memory_pool.second)});
  }
  std::stringstream compute_capability_ss;
  compute_capability_ss.setf(std::ios::fixed);
  compute_capability_ss.precision(1);
  compute_capability_ss << lserver->MinSupportedComputeCapability();
  options_table.InsertRow(std::vector<std::string>{
      "min_supported_compute_capability", compute_capability_ss.str()});
  options_table.InsertRow(std::vector<std::string>{
      "strict_readiness", std::to_string(lserver->StrictReadinessEnabled())});
  options_table.InsertRow(std::vector<std::string>{
      "exit_timeout", std::to_string(lserver->ExitTimeoutSeconds())});

  std::string options_table_string = options_table.PrintTable();
  LOG_INFO << options_table_string;

  if (!status.IsOk()) {
    if (loptions->ExitOnError()) {
      lserver->Stop(true /* force */);
      delete lserver;
      RETURN_IF_STATUS_ERROR(status);
    }

    LOG_ERROR << status.AsString();
  }

  *server = reinterpret_cast<TRITONSERVER_Server*>(lserver);
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ServerDelete(TRITONSERVER_Server* server)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);
  if (lserver != nullptr) {
    RETURN_IF_STATUS_ERROR(lserver->Stop());
  }
  delete lserver;
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ServerStop(TRITONSERVER_Server* server)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);
  if (lserver != nullptr) {
    RETURN_IF_STATUS_ERROR(lserver->Stop());
  }
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ServerPollModelRepository(TRITONSERVER_Server* server)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);
  RETURN_IF_STATUS_ERROR(lserver->PollModelRepository());
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ServerIsLive(TRITONSERVER_Server* server, bool* live)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);

  RETURN_IF_STATUS_ERROR(lserver->IsLive(live));
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ServerIsReady(TRITONSERVER_Server* server, bool* ready)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);

  RETURN_IF_STATUS_ERROR(lserver->IsReady(ready));
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ServerModelIsReady(
    TRITONSERVER_Server* server, const char* model_name,
    const int64_t model_version, bool* ready)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);

  RETURN_IF_STATUS_ERROR(
      lserver->ModelIsReady(model_name, model_version, ready));
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ServerModelBatchProperties(
    TRITONSERVER_Server* server, const char* model_name,
    const int64_t model_version, uint32_t* flags, void** voidp)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);

  if (voidp != nullptr) {
    *voidp = nullptr;
  }

  std::shared_ptr<ni::InferenceBackend> backend;
  RETURN_IF_STATUS_ERROR(
      lserver->GetInferenceBackend(model_name, model_version, &backend));

  if (backend->Config().max_batch_size() > 0) {
    *flags = TRITONSERVER_BATCH_FIRST_DIM;
  } else {
    *flags = TRITONSERVER_BATCH_UNKNOWN;
  }

  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ServerModelTransactionProperties(
    TRITONSERVER_Server* server, const char* model_name,
    const int64_t model_version, uint32_t* txn_flags, void** voidp)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);

  if (voidp != nullptr) {
    *voidp = nullptr;
  }

  *txn_flags = 0;

  std::shared_ptr<ni::InferenceBackend> backend;
  RETURN_IF_STATUS_ERROR(
      lserver->GetInferenceBackend(model_name, model_version, &backend));

  if (backend->Config().model_transaction_policy().decoupled()) {
    *txn_flags = TRITONSERVER_TXN_DECOUPLED;
  } else {
    *txn_flags = TRITONSERVER_TXN_ONE_TO_ONE;
  }

  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ServerMetadata(
    TRITONSERVER_Server* server, TRITONSERVER_Message** server_metadata)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);

  triton::common::TritonJson::Value metadata(
      triton::common::TritonJson::ValueType::OBJECT);

  // Just store string reference in JSON object since it will be
  // serialized to another buffer before lserver->Id() or
  // lserver->Version() lifetime ends.
  RETURN_IF_STATUS_ERROR(metadata.AddStringRef("name", lserver->Id().c_str()));
  RETURN_IF_STATUS_ERROR(
      metadata.AddStringRef("version", lserver->Version().c_str()));

  triton::common::TritonJson::Value extensions(
      metadata, triton::common::TritonJson::ValueType::ARRAY);
  const std::vector<const char*>& exts = lserver->Extensions();
  for (const auto ext : exts) {
    RETURN_IF_STATUS_ERROR(extensions.AppendStringRef(ext));
  }

  RETURN_IF_STATUS_ERROR(metadata.Add("extensions", std::move(extensions)));

  *server_metadata = reinterpret_cast<TRITONSERVER_Message*>(
      new ni::TritonServerMessage(metadata));
  return nullptr;  // Success
}

TRITONSERVER_Error*
TRITONSERVER_ServerModelMetadata(
    TRITONSERVER_Server* server, const char* model_name,
    const int64_t model_version, TRITONSERVER_Message** model_metadata)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);

  std::shared_ptr<ni::InferenceBackend> backend;
  RETURN_IF_STATUS_ERROR(
      lserver->GetInferenceBackend(model_name, model_version, &backend));

  std::vector<int64_t> ready_versions;
  RETURN_IF_STATUS_ERROR(
      lserver->ModelReadyVersions(model_name, &ready_versions));

  triton::common::TritonJson::Value metadata(
      triton::common::TritonJson::ValueType::OBJECT);

  // Can use string ref in this function even though model can be
  // unloaded and config becomes invalid, because TritonServeMessage
  // serializes the json when it is constructed below.
  RETURN_IF_STATUS_ERROR(metadata.AddStringRef("name", model_name));

  triton::common::TritonJson::Value versions(
      metadata, triton::common::TritonJson::ValueType::ARRAY);
  if (model_version != -1) {
    RETURN_IF_STATUS_ERROR(
        versions.AppendString(std::move(std::to_string(model_version))));
  } else {
    for (const auto v : ready_versions) {
      RETURN_IF_STATUS_ERROR(
          versions.AppendString(std::move(std::to_string(v))));
    }
  }

  RETURN_IF_STATUS_ERROR(metadata.Add("versions", std::move(versions)));

  const auto& model_config = backend->Config();
  if (!model_config.platform().empty()) {
    RETURN_IF_STATUS_ERROR(
        metadata.AddStringRef("platform", model_config.platform().c_str()));
  } else {
    RETURN_IF_STATUS_ERROR(
        metadata.AddStringRef("platform", model_config.backend().c_str()));
  }

  triton::common::TritonJson::Value inputs(
      metadata, triton::common::TritonJson::ValueType::ARRAY);
  for (const auto& io : model_config.input()) {
    triton::common::TritonJson::Value io_metadata(
        metadata, triton::common::TritonJson::ValueType::OBJECT);
    RETURN_IF_STATUS_ERROR(io_metadata.AddStringRef("name", io.name().c_str()));
    RETURN_IF_STATUS_ERROR(io_metadata.AddStringRef(
        "datatype", ni::DataTypeToProtocolString(io.data_type())));

    // Input shape. If the model supports batching then must include
    // '-1' for the batch dimension.
    triton::common::TritonJson::Value io_metadata_shape(
        metadata, triton::common::TritonJson::ValueType::ARRAY);
    if (model_config.max_batch_size() >= 1) {
      RETURN_IF_STATUS_ERROR(io_metadata_shape.AppendInt(-1));
    }
    for (const auto d : io.dims()) {
      RETURN_IF_STATUS_ERROR(io_metadata_shape.AppendInt(d));
    }
    RETURN_IF_STATUS_ERROR(
        io_metadata.Add("shape", std::move(io_metadata_shape)));

    RETURN_IF_STATUS_ERROR(inputs.Append(std::move(io_metadata)));
  }
  RETURN_IF_STATUS_ERROR(metadata.Add("inputs", std::move(inputs)));

  triton::common::TritonJson::Value outputs(
      metadata, triton::common::TritonJson::ValueType::ARRAY);
  for (const auto& io : model_config.output()) {
    triton::common::TritonJson::Value io_metadata(
        metadata, triton::common::TritonJson::ValueType::OBJECT);
    RETURN_IF_STATUS_ERROR(io_metadata.AddStringRef("name", io.name().c_str()));
    RETURN_IF_STATUS_ERROR(io_metadata.AddStringRef(
        "datatype", ni::DataTypeToProtocolString(io.data_type())));

    // Output shape. If the model supports batching then must include
    // '-1' for the batch dimension.
    triton::common::TritonJson::Value io_metadata_shape(
        metadata, triton::common::TritonJson::ValueType::ARRAY);
    if (model_config.max_batch_size() >= 1) {
      RETURN_IF_STATUS_ERROR(io_metadata_shape.AppendInt(-1));
    }
    for (const auto d : io.dims()) {
      RETURN_IF_STATUS_ERROR(io_metadata_shape.AppendInt(d));
    }
    RETURN_IF_STATUS_ERROR(
        io_metadata.Add("shape", std::move(io_metadata_shape)));

    RETURN_IF_STATUS_ERROR(outputs.Append(std::move(io_metadata)));
  }
  RETURN_IF_STATUS_ERROR(metadata.Add("outputs", std::move(outputs)));

  *model_metadata = reinterpret_cast<TRITONSERVER_Message*>(
      new ni::TritonServerMessage(metadata));
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONSERVER_ServerModelStatistics(
    TRITONSERVER_Server* server, const char* model_name,
    const int64_t model_version, TRITONSERVER_Message** model_stats)
{
#ifndef TRITON_ENABLE_STATS
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "statistics not supported");
#else

  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);

  auto model_name_string = std::string(model_name);
  std::map<std::string, std::vector<int64_t> > ready_model_versions;
  if (model_name_string.empty()) {
    RETURN_IF_STATUS_ERROR(lserver->ModelReadyVersions(&ready_model_versions));
  } else {
    std::vector<int64_t> ready_versions;
    RETURN_IF_STATUS_ERROR(
        lserver->ModelReadyVersions(model_name_string, &ready_versions));
    if (ready_versions.empty()) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          std::string(
              "requested model '" + model_name_string + "' is not available")
              .c_str());
    }

    if (model_version == -1) {
      ready_model_versions.emplace(
          model_name_string, std::move(ready_versions));
    } else {
      bool found = false;
      for (const auto v : ready_versions) {
        if (v == model_version) {
          found = true;
          break;
        }
      }
      if (found) {
        ready_model_versions.emplace(
            model_name_string, std::vector<int64_t>{model_version});
      } else {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            std::string(
                "requested model version is not available for model '" +
                model_name_string + "'")
                .c_str());
      }
    }
  }

  // Can use string ref in this function because TritonServeMessage
  // serializes the json when it is constructed below.
  triton::common::TritonJson::Value metadata(
      triton::common::TritonJson::ValueType::OBJECT);

  triton::common::TritonJson::Value model_stats_json(
      metadata, triton::common::TritonJson::ValueType::ARRAY);
  for (const auto& mv_pair : ready_model_versions) {
    for (const auto& version : mv_pair.second) {
      std::shared_ptr<ni::InferenceBackend> backend;
      RETURN_IF_STATUS_ERROR(
          lserver->GetInferenceBackend(mv_pair.first, version, &backend));
      const auto& infer_stats =
          backend->StatsAggregator().ImmutableInferStats();
      const auto& infer_batch_stats =
          backend->StatsAggregator().ImmutableInferBatchStats();

      triton::common::TritonJson::Value inference_stats(
          metadata, triton::common::TritonJson::ValueType::OBJECT);
      SetDurationStat(
          metadata, inference_stats, "success", infer_stats.success_count_,
          infer_stats.request_duration_ns_);
      SetDurationStat(
          metadata, inference_stats, "fail", infer_stats.failure_count_,
          infer_stats.failure_duration_ns_);
      SetDurationStat(
          metadata, inference_stats, "queue", infer_stats.success_count_,
          infer_stats.queue_duration_ns_);
      SetDurationStat(
          metadata, inference_stats, "compute_input",
          infer_stats.success_count_, infer_stats.compute_input_duration_ns_);
      SetDurationStat(
          metadata, inference_stats, "compute_infer",
          infer_stats.success_count_, infer_stats.compute_infer_duration_ns_);
      SetDurationStat(
          metadata, inference_stats, "compute_output",
          infer_stats.success_count_, infer_stats.compute_output_duration_ns_);

      triton::common::TritonJson::Value batch_stats(
          metadata, triton::common::TritonJson::ValueType::ARRAY);
      for (const auto& batch : infer_batch_stats) {
        triton::common::TritonJson::Value batch_stat(
            metadata, triton::common::TritonJson::ValueType::OBJECT);
        RETURN_IF_STATUS_ERROR(batch_stat.AddUInt("batch_size", batch.first));
        SetDurationStat(
            metadata, batch_stat, "compute_input", batch.second.count_,
            batch.second.compute_input_duration_ns_);
        SetDurationStat(
            metadata, batch_stat, "compute_infer", batch.second.count_,
            batch.second.compute_infer_duration_ns_);
        SetDurationStat(
            metadata, batch_stat, "compute_output", batch.second.count_,
            batch.second.compute_output_duration_ns_);
        RETURN_IF_STATUS_ERROR(batch_stats.Append(std::move(batch_stat)));
      }

      triton::common::TritonJson::Value model_stat(
          metadata, triton::common::TritonJson::ValueType::OBJECT);
      RETURN_IF_STATUS_ERROR(
          model_stat.AddStringRef("name", mv_pair.first.c_str()));
      RETURN_IF_STATUS_ERROR(
          model_stat.AddString("version", std::move(std::to_string(version))));

      RETURN_IF_STATUS_ERROR(model_stat.AddUInt(
          "last_inference", backend->StatsAggregator().LastInferenceMs()));
      RETURN_IF_STATUS_ERROR(model_stat.AddUInt(
          "inference_count", backend->StatsAggregator().InferenceCount()));
      RETURN_IF_STATUS_ERROR(model_stat.AddUInt(
          "execution_count", backend->StatsAggregator().ExecutionCount()));

      RETURN_IF_STATUS_ERROR(
          model_stat.Add("inference_stats", std::move(inference_stats)));
      RETURN_IF_STATUS_ERROR(
          model_stat.Add("batch_stats", std::move(batch_stats)));
      RETURN_IF_STATUS_ERROR(model_stats_json.Append(std::move(model_stat)));
    }
  }

  RETURN_IF_STATUS_ERROR(
      metadata.Add("model_stats", std::move(model_stats_json)));
  *model_stats = reinterpret_cast<TRITONSERVER_Message*>(
      new ni::TritonServerMessage(metadata));

  return nullptr;  // success

#endif  // TRITON_ENABLE_STATS
}

TRITONSERVER_Error*
TRITONSERVER_ServerModelConfig(
    TRITONSERVER_Server* server, const char* model_name,
    const int64_t model_version, const uint32_t config_version,
    TRITONSERVER_Message** model_config)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);

  std::shared_ptr<ni::InferenceBackend> backend;
  RETURN_IF_STATUS_ERROR(
      lserver->GetInferenceBackend(model_name, model_version, &backend));

  std::string model_config_json;
  RETURN_IF_STATUS_ERROR(ni::ModelConfigToJson(
      backend->Config(), config_version, &model_config_json));

  *model_config = reinterpret_cast<TRITONSERVER_Message*>(
      new ni::TritonServerMessage(std::move(model_config_json)));

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONSERVER_ServerModelIndex(
    TRITONSERVER_Server* server, uint32_t flags,
    TRITONSERVER_Message** repository_index)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);

  const bool ready_only = ((flags & TRITONSERVER_INDEX_FLAG_READY) != 0);

  std::vector<ni::ModelRepositoryManager::ModelIndex> index;
  RETURN_IF_STATUS_ERROR(lserver->RepositoryIndex(ready_only, &index));

  // Can use string ref in this function because TritonServeMessage
  // serializes the json when it is constructed below.
  triton::common::TritonJson::Value repository_index_json(
      triton::common::TritonJson::ValueType::ARRAY);

  for (const auto& in : index) {
    triton::common::TritonJson::Value model_index(
        repository_index_json, triton::common::TritonJson::ValueType::OBJECT);
    RETURN_IF_STATUS_ERROR(model_index.AddStringRef("name", in.name_.c_str()));
    if (!in.name_only_) {
      if (in.version_ >= 0) {
        RETURN_IF_STATUS_ERROR(model_index.AddString(
            "version", std::move(std::to_string(in.version_))));
      }
      RETURN_IF_STATUS_ERROR(model_index.AddStringRef(
          "state", ni::ModelReadyStateString(in.state_).c_str()));
      if (!in.reason_.empty()) {
        RETURN_IF_STATUS_ERROR(
            model_index.AddStringRef("reason", in.reason_.c_str()));
      }
    }

    RETURN_IF_STATUS_ERROR(
        repository_index_json.Append(std::move(model_index)));
  }

  *repository_index = reinterpret_cast<TRITONSERVER_Message*>(
      new ni::TritonServerMessage(repository_index_json));

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONSERVER_ServerLoadModel(
    TRITONSERVER_Server* server, const char* model_name)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);

  RETURN_IF_STATUS_ERROR(lserver->LoadModel(std::string(model_name)));

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONSERVER_ServerUnloadModel(
    TRITONSERVER_Server* server, const char* model_name)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);

  RETURN_IF_STATUS_ERROR(lserver->UnloadModel(
      std::string(model_name), false /* unload_dependents */));

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONSERVER_ServerUnloadModelAndDependents(
    TRITONSERVER_Server* server, const char* model_name)
{
  {
    ni::InferenceServer* lserver =
        reinterpret_cast<ni::InferenceServer*>(server);

    RETURN_IF_STATUS_ERROR(lserver->UnloadModel(
        std::string(model_name), true /* unload_dependents */));

    return nullptr;  // success
  }
}

TRITONSERVER_Error*
TRITONSERVER_ServerMetrics(
    TRITONSERVER_Server* server, TRITONSERVER_Metrics** metrics)
{
#ifdef TRITON_ENABLE_METRICS
  TritonServerMetrics* lmetrics = new TritonServerMetrics();
  *metrics = reinterpret_cast<TRITONSERVER_Metrics*>(lmetrics);
  return nullptr;  // Success
#else
  *metrics = nullptr;
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "metrics not supported");
#endif  // TRITON_ENABLE_METRICS
}

TRITONSERVER_Error*
TRITONSERVER_ServerInferAsync(
    TRITONSERVER_Server* server,
    TRITONSERVER_InferenceRequest* inference_request,
    TRITONSERVER_InferenceTrace* trace)
{
  ni::InferenceServer* lserver = reinterpret_cast<ni::InferenceServer*>(server);
  ni::InferenceRequest* lrequest =
      reinterpret_cast<ni::InferenceRequest*>(inference_request);

  RETURN_IF_STATUS_ERROR(lrequest->PrepareForInference());

  // Set the trace object in the request so that activity associated
  // with the request can be recorded as the request flows through
  // Triton.
  if (trace != nullptr) {
#ifdef TRITON_ENABLE_TRACING
    ni::InferenceTrace* ltrace = reinterpret_cast<ni::InferenceTrace*>(trace);
    ltrace->SetModelName(lrequest->ModelName());
    ltrace->SetModelVersion(lrequest->ActualModelVersion());

    std::unique_ptr<ni::InferenceTrace> utrace(ltrace);
    lrequest->SetTrace(std::move(utrace));
#else
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED, "inference tracing not supported");
#endif  // TRITON_ENABLE_TRACING
  }

  // We wrap the request in a unique pointer to ensure that it flows
  // through inferencing with clear ownership.
  std::unique_ptr<ni::InferenceRequest> ureq(lrequest);

  // Run inference...
  ni::Status status = lserver->InferAsync(ureq);

  // If there is an error then must explicitly release any trace
  // object associated with the inference request above.
#ifdef TRITON_ENABLE_TRACING
  if (!status.IsOk()) {
    std::unique_ptr<ni::InferenceTrace>* trace = ureq->MutableTrace();
    if (*trace != nullptr) {
      ni::InferenceTrace::Release(std::move(*trace));
    }
  }
#endif  // TRITON_ENABLE_TRACING

  // If there is an error then ureq will still have 'lrequest' and we
  // must release it from unique_ptr since the caller should retain
  // ownership when there is error. If there is not an error then ureq
  // == nullptr and so this release is a nop.
  ureq.release();

  RETURN_IF_STATUS_ERROR(status);
  return nullptr;  // Success
}

#ifdef __cplusplus
}
#endif
