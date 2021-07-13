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

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <vector>

#include "model_config.pb.h"
#include "src/backends/backend/triton_model.h"
#include "src/backends/backend/triton_model_instance.h"
#include "src/core/status.h"

namespace nvidia { namespace inferenceserver {

// Limits the rate at which requests are dispatched from the scheduler
class RateLimiter {
 private:
  class ModelContext;

 public:
  class ModelInstance;
  class Payload;


  using StandardReleaseFunc = std::function<void(ModelInstance*)>;
  using StandardScheduleFunc = std::function<void(ModelInstance*)>;
  using StandardStageFunc = std::function<void(ModelInstance*)>;
  using RateLimiterConfig = inference::ModelRateLimiter;

  using ResourceMap = std::map<int, std::map<std::string, size_t>>;
  enum RESOURCE_KIND_KEY {
    // Key for holding global resources
    GLOBAL_RESOURCE_KEY = -2,
    // Key for holding resources per each device
    PER_DEVICE_RESOURCE_KEY = -1
  };


  /// Creates a rate limiter object which will funnel the requests to
  /// the model instances. A typical lifetime of the model instance within
  /// RateLimiter transition from available -> staged -> allocated -> available.
  /// The transition from available to staged occurs when a request is
  /// registered for the model. Depending upon the resource availabilty and
  /// priority, the RateLimiter will transition an instance to allocated state
  /// at some point in the future. The staged state is skipped when
  /// configured to ignore the resource constraints. The cycle in this case
  /// will be available -> allocated -> available.
  /// \param ignore_resources_and_priority Whether or not to ignore resource
  /// constraints and cross-model priority. An available instance is directly
  /// allocated when true.
  /// \param resource_map The map to the available resource count provided
  /// explicitly.
  /// \return Status object indicating success or failure.
  static Status Create(
      const bool ignore_resources_and_priority, const ResourceMap& resource_map,
      std::unique_ptr<RateLimiter>* rate_limiter);

  /// Registers the model instance with the rate limiter.
  /// \param instance The pointer to the TritonModelInstance object to register
  /// with the rate limiter.
  /// \param rate_limiter_config The rate limiter configuration associated with
  /// the model instance.
  Status RegisterModelInstance(
      TritonModelInstance* instance,
      const RateLimiterConfig& rate_limiter_config);

  /// Remove model from the set of models being managed by the rate limiter.
  /// \param model The pointer to TritonModel object to be removed.
  /// \return Status object indicating success or failure.
  Status UnregisterModel(const TritonModel* model);

  void InitializePayloadQueues(const TritonModelInstance* instance);
  bool PayloadSlotAvailable(const TritonModel* model);

  Status EnqueuePayload(
      const TritonModel* model, std::shared_ptr<Payload> payload);

  void DequeuePayload(
      std::deque<TritonModelInstance*>& instance,
      std::shared_ptr<Payload>* payload);

  /// Requests one of the available model instance. In future, when the
  /// conditions are met, the callback will be invoked and a pointer to
  /// allocated RateLimiter::ModelInstance object will be exposed as a
  /// parameter. The user must ensure RateLimiter::ModelInstance::Release
  /// gets called on the instance once the inference request is complete
  /// so that the instance and its resources are returned to the available
  /// pool. Also, note the callback should be a light-weight call and
  /// must not itself invoke the inference execution but just be used
  /// as a signal to proceed with the execution.
  /// \param OnSchedule The callback function to be called when scheduling.
  /// \param model The TritonModel object pointer to be used for running the
  /// inference.
  /// \param instance The TritonModelInstance object pointer to be used for
  /// running the inference. The default value is nullptr which means that an
  /// instance with highest priority will be selected for the execution.
  /// \return Status object indicating success or failure.
  Status RequestModelInstance(
      const StandardScheduleFunc& OnSchedule, const TritonModel* model,
      TritonModelInstance* instance = nullptr);

  /// Whether or not to ignore the resource configurations and priority settings
  /// for the rate limiter.
  bool IgnoreResourcesAndPriority() { return ignore_resources_and_priority_; }

  class Payload {
   public:
    enum Operation { INFER_RUN = 0, INIT = 1, WARM_UP = 2, EXIT = 3 };
    enum State {
      UNINITIALIZED = 0,
      READY = 1,
      REQUESTED = 2,
      SCHEDULED = 3,
      EXECUTING = 4,
      RELEASED = 5
    };

    Payload();
    void Reset(
        const Operation op_type, TritonModelInstance* instance = nullptr);
    Operation GetOpType() { return op_type_; }
    std::mutex* GetExecMutex() { return exec_mu_.get(); }
    size_t BatchSize() { return requests_.size(); }
    void ReserveRequests(size_t size);
    void AddRequest(std::unique_ptr<InferenceRequest> request);
    void SetCallback(std::function<void()> OnCallback);
    void Callback();
    void SetInstance(TritonModelInstance* model_instance);
    TritonModelInstance* GetInstance() { return instance_; }

    State GetState() { return state_; }
    void SetState(State state);
    void Execute(bool* should_exit);
    Status Wait();
    void Release();

   private:
    Operation op_type_;
    std::vector<std::unique_ptr<InferenceRequest>> requests_;
    std::function<void()> OnCallback_;
    TritonModelInstance* instance_;
    State state_;
    std::unique_ptr<std::promise<Status>> status_;
    std::unique_ptr<std::mutex> exec_mu_;
  };


  std::shared_ptr<Payload> GetPayload(
      const Payload::Operation op_type,
      TritonModelInstance* instance = nullptr);
  void PayloadRelease(std::shared_ptr<Payload>& payload);

  // Holds the state of the model instance.
  class ModelInstance {
   public:
    friend class RateLimiter;
    friend class ResourceManager;
    enum State { AVAILABLE, STAGED, ALLOCATED, REMOVED };

    /// Should be called when the request on the model instance is
    /// complete. This function releases the resources allocated to
    /// the model instance and sends the instance into the available
    /// pool so that it can serve other requests.
    void Release();

    /// Returns the raw triton instance
    TritonModelInstance* RawInstance() const { return triton_model_instance_; }

   private:
    ModelInstance(
        TritonModelInstance* triton_model_instance, ModelContext* model_context,
        const RateLimiterConfig& rate_limiter_config, StandardStageFunc OnStage,
        StandardReleaseFunc OnRelease);

    const RateLimiterConfig* GetRateLimiterConfig() const
    {
      return &rate_limiter_config_;
    }
    void MarkAvailable();
    double ScaledPriority();
    Status Stage(StandardScheduleFunc OnSchedule);
    Status Allocate();
    Status DirectAllocate(StandardScheduleFunc OnSchedule);
    void RequestRemoval();
    void WaitForRemoval();

    TritonModelInstance* triton_model_instance_;
    ModelContext* model_context_;
    RateLimiterConfig rate_limiter_config_;
    StandardStageFunc OnStage_;
    StandardReleaseFunc OnRelease_;
    bool executed_;
    std::atomic<uint64_t> exec_count_;

    State state_;
    bool removal_in_progress_;
    std::mutex state_mtx_;

    StandardScheduleFunc OnSchedule_;

    std::condition_variable cv_;
  };

 private:
  RateLimiter(
      const bool ignore_resources_and_priority,
      const ResourceMap& resource_map);

  void OnStage(ModelInstance* instance_ptr);
  void OnRelease(ModelInstance* instance_ptr);
  void AttemptAllocation();

  class ScaledPriorityComparator {
   public:
    bool operator()(ModelInstance* a, ModelInstance* b)
    {
      return a->ScaledPriority() > b->ScaledPriority();
    }
  };

  using PriorityQueue = std::priority_queue<
      ModelInstance*, std::vector<ModelInstance*>, ScaledPriorityComparator>;

  // Holds the active context to a model
  class ModelContext {
   public:
    ModelContext();

    Status EnqueueModelInstanceRequest(
        const StandardScheduleFunc& OnSchedule,
        TritonModelInstance* triton_model_instance);
    void AddAvailableInstance(ModelInstance* instance);
    void StageInstanceIfAvailable();
    void AllocateInstanceIfAvailable();
    void AddSpecificRequestQueue();
    bool ContainsPendingRequests(int32_t index);
    void RequestRemoval();
    bool isRemovalInProgress() { return removal_in_progress_; }

   private:
    bool removal_in_progress_;

    // Queue holding pending scheduling request
    std::queue<StandardScheduleFunc> generic_request_queue_;
    std::vector<std::queue<StandardScheduleFunc>> specific_request_queues_;
    std::recursive_mutex request_queue_mtx_;

    // The set of instances that are available at the moment
    PriorityQueue avbl_instances_;
    std::recursive_mutex avbl_instances_mtx_;
  };

  // Manages and keep track of resource allocation to the model instances.
  class ResourceManager {
   public:
    static Status Create(
        const ResourceMap& resource_map,
        std::unique_ptr<ResourceManager>* resource_manager);
    void AddModelInstance(const RateLimiter::ModelInstance* instance);
    Status RemoveModelInstance(const RateLimiter::ModelInstance* instance);
    Status UpdateResourceLimits();
    bool AllocateResources(const RateLimiter::ModelInstance* instance);
    Status ReleaseResources(const RateLimiter::ModelInstance* instance);

   private:
    ResourceManager(const ResourceMap& resource_map);
    Status ValidateMaxResources();
    Status ParseAndValidateExplicitResources();

    ResourceMap explicit_max_resources_;

    std::map<const RateLimiter::ModelInstance*, ResourceMap> model_resources_;
    std::mutex model_resources_mtx_;

    ResourceMap max_resources_;
    std::mutex max_resources_mtx_;

    ResourceMap allocated_resources_;
    std::mutex allocated_resources_mtx_;
  };

  bool ignore_resources_and_priority_;

  // Instances for the models
  std::map<const TritonModel*, std::vector<std::shared_ptr<ModelInstance>>>
      model_instances_;
  std::mutex model_instances_mtx_;

  // Running context of the models
  std::map<const TritonModel*, ModelContext> model_contexts_;
  std::mutex model_contexts_mtx_;

  // Holds the model instances that have been staged
  PriorityQueue staged_instances_;
  std::recursive_mutex staged_instances_mtx_;

  // Manager to keep track of the resource allocations
  std::unique_ptr<ResourceManager> resource_manager_;

  // Mutex to serialize Payload allocation
  std::mutex alloc_mu_;

  // Keep some number of Payload objects for reuse to avoid the overhead
  // of creating a Payload for every new request.
  const size_t max_payload_bucket_count_;
  std::vector<std::shared_ptr<Payload>> payload_bucket_;

  struct PayloadQueue {
    std::deque<std::shared_ptr<Payload>> queue_;
    std::map<
        const TritonModelInstance*,
        std::unique_ptr<std::deque<std::shared_ptr<Payload>>>>
        specific_queues_;
    std::mutex mu_;
    std::condition_variable cv_;
  };
  std::map<const TritonModel*, std::unique_ptr<PayloadQueue>> payload_queues_;
};

}}  // namespace nvidia::inferenceserver
