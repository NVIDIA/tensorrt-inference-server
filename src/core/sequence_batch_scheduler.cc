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

#include "src/core/sequence_batch_scheduler.h"

#ifndef _WIN32
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif
#include "src/core/constants.h"
#include "src/core/dynamic_batch_scheduler.h"
#include "src/core/logging.h"
#include "src/core/model_config_utils.h"
#include "src/core/server.h"

namespace nvidia { namespace inferenceserver {

Status
SequenceBatchScheduler::Create(
    TritonModel* model,
    const std::unordered_map<std::string, bool>& enforce_equal_shape_tensors,
    std::unique_ptr<Scheduler>* scheduler)
{
  std::unique_ptr<SequenceBatchScheduler> sched(new SequenceBatchScheduler());

  // For debugging and testing,
  const char* dstr = getenv("TRITONSERVER_BACKLOG_DELAY_SCHEDULER");
  sched->backlog_delay_cnt_ = 0;
  if (dstr != nullptr) {
    sched->backlog_delay_cnt_ = atoi(dstr);
    LOG_INFO << "Delaying scheduler until " << sched->backlog_delay_cnt_
             << " backlog queued requests...";
  }

  auto instance_count = model->Instances().size();
  sched->queue_request_cnts_.resize(instance_count, 0);

  auto config = model->Config();

  // Max sequence idle...
  sched->max_sequence_idle_microseconds_ =
      config.sequence_batching().max_sequence_idle_microseconds();

  // Get the number of candidate sequence slots to allow for each
  // runner. This is at least 1 even if the model doesn't support
  // batching.
  const size_t model_batch_size = std::max(1, config.max_batch_size());
  size_t seq_slot_cnt = model_batch_size;
  if (config.sequence_batching().has_oldest()) {
    seq_slot_cnt =
        config.sequence_batching().oldest().max_candidate_sequences();
  }

  // Based on the model configuration create input tensors for control
  // signals indicating sequence start, sequence continue, and
  // sequence not ready.
  std::shared_ptr<ControlInputs> start;
  std::shared_ptr<ControlInputs> end;
  std::shared_ptr<ControlInputs> startend;
  std::shared_ptr<ControlInputs> cont;
  std::shared_ptr<ControlInputs> notready;
  RETURN_IF_ERROR(sched->CreateBooleanControlTensors(
      config, &start, &end, &startend, &cont, &notready));

  // Create one SequenceBatch object for each requested runner. The
  // SequenceBatch object has a thread that manages the batch of
  // requests.
  const auto& instances = model->Instances();
  uint32_t index = 0;
  for (const auto& instance : instances) {
    bool init_state;
    std::shared_ptr<SequenceBatch> sb;

    // Create the SequenceBatch derivative that handles the requested
    // scheduling strategy.
    if (config.sequence_batching().has_oldest()) {
      sb.reset(new OldestSequenceBatch(
          sched.get(), index, seq_slot_cnt, instance.get(),
          enforce_equal_shape_tensors, start, end, startend, cont, notready,
          &init_state));
    } else {
      sb.reset(new DirectSequenceBatch(
          sched.get(), index, seq_slot_cnt, instance.get(),
          enforce_equal_shape_tensors, start, end, startend, cont, notready,
          &init_state));
    }

    if (init_state) {
      sched->batchers_.push_back(sb);
      // All sequence slots in the batcher are initially ready for a
      // new sequence.
      for (size_t b = 0; b < seq_slot_cnt; ++b) {
        sched->ready_batcher_seq_slots_.push(
            SequenceBatchScheduler::BatcherSequenceSlot(index, b));
      }
    }
    ++index;
  }
  if (sched->batchers_.empty()) {
    return Status(
        Status::Code::INTERNAL,
        "Initialization failed for all sequence-batch scheduler threads");
  }

  // Create a reaper thread that watches for idle sequences. Run the
  // reaper a lower priority.
  SequenceBatchScheduler* raw = sched.release();

  raw->reaper_thread_exit_ = false;
  raw->reaper_thread_.reset(
      new std::thread([raw]() { raw->ReaperThread(10 /* nice */); }));

  scheduler->reset(raw);

  return Status::Success;
}

SequenceBatchScheduler::~SequenceBatchScheduler()
{
  // Signal the reaper thread to exit...
  {
    std::unique_lock<std::mutex> lock(mu_);
    reaper_thread_exit_ = true;
  }

  reaper_cv_.notify_one();
  if ((reaper_thread_ != nullptr) && reaper_thread_->joinable()) {
    reaper_thread_->join();
  }
}

namespace {

Status
GetBooleanOverrideInputs(
    const std::string& tensor_name, const bool support_batching,
    const inference::DataType tensor_datatype, const float fp32_false_value,
    const float fp32_true_value, const int32_t int32_false_value,
    const int32_t int32_true_value,
    std::shared_ptr<InferenceRequest::Input>* true_override,
    std::shared_ptr<InferenceRequest::Input>* false_override)
{
  TRITONSERVER_MemoryType memory_type;
  int64_t memory_type_id;

  const std::vector<int64_t> tensor_shape{1};
  std::vector<int64_t> tensor_shape_with_batch_dim{1};
  if (support_batching) {
    tensor_shape_with_batch_dim.push_back(1);
  }
  const size_t size_p = GetDataTypeByteSize(tensor_datatype);

  auto true_p =
      std::make_shared<AllocatedMemory>(size_p, TRITONSERVER_MEMORY_CPU, 0);
  char* true_p_ptr = true_p->MutableBuffer(&memory_type, &memory_type_id);
  if ((true_p_ptr == nullptr) ||
      ((memory_type != TRITONSERVER_MEMORY_CPU) &&
       (memory_type != TRITONSERVER_MEMORY_CPU_PINNED)) ||
      (memory_type_id != 0)) {
    return Status(
        Status::Code::INTERNAL,
        "failed to allocate sequence control signal in CPU memory");
  }

  auto false_p =
      std::make_shared<AllocatedMemory>(size_p, TRITONSERVER_MEMORY_CPU, 0);
  char* false_p_ptr = false_p->MutableBuffer(&memory_type, &memory_type_id);
  if ((false_p_ptr == nullptr) ||
      ((memory_type != TRITONSERVER_MEMORY_CPU) &&
       (memory_type != TRITONSERVER_MEMORY_CPU_PINNED)) ||
      (memory_type_id != 0)) {
    return Status(
        Status::Code::INTERNAL,
        "failed to allocate sequence control signal in CPU memory");
  }

  if (tensor_datatype == inference::DataType::TYPE_INT32) {
    *(reinterpret_cast<int32_t*>(true_p_ptr)) = int32_true_value;
    *(reinterpret_cast<int32_t*>(false_p_ptr)) = int32_false_value;
  } else {
    *(reinterpret_cast<float*>(true_p_ptr)) = fp32_true_value;
    *(reinterpret_cast<float*>(false_p_ptr)) = fp32_false_value;
  }

  auto ltrue_override = std::make_shared<InferenceRequest::Input>(
      tensor_name, tensor_datatype, tensor_shape);
  *ltrue_override->MutableShape() = ltrue_override->OriginalShape();
  *ltrue_override->MutableShapeWithBatchDim() = tensor_shape_with_batch_dim;
  RETURN_IF_ERROR(ltrue_override->SetData(true_p));

  auto lfalse_override = std::make_shared<InferenceRequest::Input>(
      tensor_name, tensor_datatype, tensor_shape);
  *lfalse_override->MutableShape() = lfalse_override->OriginalShape();
  *lfalse_override->MutableShapeWithBatchDim() = tensor_shape_with_batch_dim;
  RETURN_IF_ERROR(lfalse_override->SetData(false_p));

  *true_override = std::move(ltrue_override);
  *false_override = std::move(lfalse_override);

  return Status::Success;
}

}  // namespace

Status
SequenceBatchScheduler::CreateBooleanControlTensors(
    const inference::ModelConfig& config,
    std::shared_ptr<ControlInputs>* start_input_overrides,
    std::shared_ptr<ControlInputs>* end_input_overrides,
    std::shared_ptr<ControlInputs>* startend_input_overrides,
    std::shared_ptr<ControlInputs>* continue_input_overrides,
    std::shared_ptr<ControlInputs>* notready_input_overrides)
{
  // Currently only batch-size 1 requests are supported so only need
  // to provide control vectors of that size.
  *start_input_overrides = std::make_shared<ControlInputs>();
  *end_input_overrides = std::make_shared<ControlInputs>();
  *startend_input_overrides = std::make_shared<ControlInputs>();
  *continue_input_overrides = std::make_shared<ControlInputs>();
  *notready_input_overrides = std::make_shared<ControlInputs>();

  std::string tensor_name;
  inference::DataType tensor_datatype;
  int32_t int32_false_value, int32_true_value;
  float fp32_false_value, fp32_true_value;

  // START, optional
  {
    RETURN_IF_ERROR(GetBooleanSequenceControlProperties(
        config.sequence_batching(), config.name(),
        inference::ModelSequenceBatching::Control::CONTROL_SEQUENCE_START,
        false /* required */, &tensor_name, &tensor_datatype, &fp32_false_value,
        &fp32_true_value, &int32_false_value, &int32_true_value));
    if (!tensor_name.empty()) {
      std::shared_ptr<InferenceRequest::Input> true_override;
      std::shared_ptr<InferenceRequest::Input> false_override;

      RETURN_IF_ERROR(GetBooleanOverrideInputs(
          tensor_name, config.max_batch_size() != 0, tensor_datatype,
          fp32_false_value, fp32_true_value, int32_false_value,
          int32_true_value, &true_override, &false_override));

      (*start_input_overrides)->emplace_back(true_override);
      (*end_input_overrides)->emplace_back(false_override);
      (*startend_input_overrides)->emplace_back(true_override);
      (*continue_input_overrides)->emplace_back(false_override);
      (*notready_input_overrides)->emplace_back(false_override);
    }
  }

  // END, optional
  {
    RETURN_IF_ERROR(GetBooleanSequenceControlProperties(
        config.sequence_batching(), config.name(),
        inference::ModelSequenceBatching::Control::CONTROL_SEQUENCE_END,
        false /* required */, &tensor_name, &tensor_datatype, &fp32_false_value,
        &fp32_true_value, &int32_false_value, &int32_true_value));
    if (!tensor_name.empty()) {
      std::shared_ptr<InferenceRequest::Input> true_override;
      std::shared_ptr<InferenceRequest::Input> false_override;

      RETURN_IF_ERROR(GetBooleanOverrideInputs(
          tensor_name, config.max_batch_size() != 0, tensor_datatype,
          fp32_false_value, fp32_true_value, int32_false_value,
          int32_true_value, &true_override, &false_override));

      (*start_input_overrides)->emplace_back(false_override);
      (*end_input_overrides)->emplace_back(true_override);
      (*startend_input_overrides)->emplace_back(true_override);
      (*continue_input_overrides)->emplace_back(false_override);
      (*notready_input_overrides)->emplace_back(false_override);
    }
  }

  // READY, optional
  {
    RETURN_IF_ERROR(GetBooleanSequenceControlProperties(
        config.sequence_batching(), config.name(),
        inference::ModelSequenceBatching::Control::CONTROL_SEQUENCE_READY,
        false /* required */, &tensor_name, &tensor_datatype, &fp32_false_value,
        &fp32_true_value, &int32_false_value, &int32_true_value));
    if (!tensor_name.empty()) {
      std::shared_ptr<InferenceRequest::Input> true_override;
      std::shared_ptr<InferenceRequest::Input> false_override;

      RETURN_IF_ERROR(GetBooleanOverrideInputs(
          tensor_name, config.max_batch_size() != 0, tensor_datatype,
          fp32_false_value, fp32_true_value, int32_false_value,
          int32_true_value, &true_override, &false_override));

      (*start_input_overrides)->emplace_back(true_override);
      (*end_input_overrides)->emplace_back(true_override);
      (*startend_input_overrides)->emplace_back(true_override);
      (*continue_input_overrides)->emplace_back(true_override);
      (*notready_input_overrides)->emplace_back(false_override);
    }
  }

  return Status::Success;
}

Status
SequenceBatchScheduler::Enqueue(std::unique_ptr<InferenceRequest>& irequest)
{
  // Queue timer starts at the beginning of the queueing and
  // scheduling process
  irequest->CaptureQueueStartNs();
  INFER_TRACE_ACTIVITY(
      irequest->Trace(), TRITONSERVER_TRACE_QUEUE_START,
      irequest->QueueStartNs());

  // For now the request must have batch-size 1 since the sequence
  // batcher does not yet support requests that are statically
  // batched.
  if (irequest->BatchSize() > 1) {
    return Status(
        Status::Code::INVALID_ARG,
        "inference request to model '" + irequest->ModelName() +
            "' must specify batch-size 1 due to requirements of sequence "
            "batcher");
  }

  // A request must have a correlation ID to be processed correctly by
  // this scheduler. A value of 0 (zero) indicates that the request
  // doesn't have a correlation ID.
  const uint64_t correlation_id = irequest->CorrelationId();
  if (correlation_id == 0) {
    return Status(
        Status::Code::INVALID_ARG,
        "inference request to model '" + irequest->ModelName() +
            "' must specify a non-zero correlation ID");
  }

  BatcherSequenceSlot* target = nullptr;

  const bool seq_start =
      ((irequest->Flags() & TRITONSERVER_REQUEST_FLAG_SEQUENCE_START) != 0);
  const bool seq_end =
      ((irequest->Flags() & TRITONSERVER_REQUEST_FLAG_SEQUENCE_END) != 0);

  std::unique_lock<std::mutex> lock(mu_);

  auto sb_itr = sequence_to_batcherseqslot_map_.find(correlation_id);
  auto bl_itr = sequence_to_backlog_map_.find(correlation_id);

  // If this request is not starting a new sequence its correlation ID
  // should already be known with a target in either a sequence slot
  // or in the backlog. If it doesn't then the sequence wasn't started
  // correctly or there has been a correlation ID conflict. In either
  // case fail this request.
  if (!seq_start && (sb_itr == sequence_to_batcherseqslot_map_.end()) &&
      (bl_itr == sequence_to_backlog_map_.end())) {
    return Status(
        Status::Code::INVALID_ARG,
        "inference request for sequence " + std::to_string(correlation_id) +
            " to model '" + irequest->ModelName() +
            "' must specify the START flag on the first request of the "
            "sequence");
  }

  // Record the timestamp of this request for the correlation ID. The
  // reaper thread will check to make sure that
  // max_sequence_idle_microseconds value is not exceed for any
  // sequence, and if it is it will release the sequence slot (if any)
  // allocated to that sequence.
  {
    uint64_t now_us = std::chrono::duration_cast<std::chrono::microseconds>(
                          std::chrono::steady_clock::now().time_since_epoch())
                          .count();
    correlation_id_timestamps_[correlation_id] = now_us;
  }

  // If this request starts a new sequence but the correlation ID
  // already has an in-progress sequence then that previous sequence
  // did not end correctly, or there is a correlation ID conflict. In
  // this case we continue the new sequence (in either backlog or
  // sequence slot). It is ok for a backlog/slot to have multiple
  // starts... as long as it has a single end. The previous sequence
  // that was not correctly ended will have its existing requests
  // handled and then the new sequence will start.
  if (seq_start && ((sb_itr != sequence_to_batcherseqslot_map_.end()) ||
                    (bl_itr != sequence_to_backlog_map_.end()))) {
    LOG_WARNING
        << "sequence " << correlation_id << " for model '"
        << irequest->ModelName()
        << "' has a conflict. The previous sequence did not end before this "
           "sequence start. Previous sequence will be terminated early.";
  }

  // This request already has an assigned slot...
  if (sb_itr != sequence_to_batcherseqslot_map_.end()) {
    target = &sb_itr->second;
  }
  // This request already has a queue in the backlog...
  else if (bl_itr != sequence_to_backlog_map_.end()) {
    LOG_VERBOSE(1) << "Enqueuing CORRID " << correlation_id
                   << " into existing backlog: " << irequest->ModelName();

    bl_itr->second->emplace_back(std::move(irequest));

    // If the sequence is ending then forget correlation ID
    // connection to this backlog queue. If another sequence starts
    // with the same correlation ID it will be collected in another
    // backlog queue.
    if (seq_end) {
      sequence_to_backlog_map_.erase(bl_itr);
    }
    return Status::Success;
  }
  // This request does not have an assigned backlog or sequence
  // slot. By the above checks it must be starting. If there is a free
  // sequence slot available then assign this sequence to that slot...
  else if (!ready_batcher_seq_slots_.empty()) {
    target = &sequence_to_batcherseqslot_map_[correlation_id];
    *target = ready_batcher_seq_slots_.top();
    ready_batcher_seq_slots_.pop();
  }
  // Last option is to assign this request to the backlog...
  else {
    LOG_VERBOSE(1) << "Enqueuing CORRID " << correlation_id
                   << " into new backlog: " << irequest->ModelName();

    auto backlog =
        std::make_shared<std::deque<std::unique_ptr<InferenceRequest>>>();
    backlog_queues_.push_back(backlog);
    backlog->emplace_back(std::move(irequest));
    if (!seq_end) {
      sequence_to_backlog_map_[correlation_id] = std::move(backlog);
    }
    return Status::Success;
  }

  // Need to grab the target contents before the erase below since
  // that can free it.
  const size_t batcher_idx = target->batcher_idx_;
  const uint32_t seq_slot = target->seq_slot_;

  // At this point the request has been assigned to a sequence
  // slot. If the sequence is ending then stop tracking the
  // correlation.
  if (seq_end) {
    sequence_to_batcherseqslot_map_.erase(correlation_id);
  }

  // Enqueue request into batcher and sequence slot.  Don't hold the
  // lock while enqueuing in a specific batcher.
  lock.unlock();

  LOG_VERBOSE(1) << "Enqueuing CORRID " << correlation_id << " into batcher "
                 << batcher_idx << ", sequence slot " << seq_slot << ": "
                 << irequest->ModelName();

  batchers_[batcher_idx]->Enqueue(seq_slot, correlation_id, irequest);

  return Status::Success;
}

uint64_t
SequenceBatchScheduler::ReleaseSequenceSlot(
    const BatcherSequenceSlot& batcher_seq_slot,
    std::deque<std::unique_ptr<InferenceRequest>>* requests)
{
  std::unique_lock<std::mutex> lock(mu_);

  // If there is a backlogged sequence and it is requested, return it
  // so that it can use the newly available sequence slot.
  if (!backlog_queues_.empty()) {
    auto& backlog = backlog_queues_.front();
    *requests = std::move(*backlog);
    backlog_queues_.pop_front();
    if (!requests->empty()) {  // should never be empty...
      const auto& irequest = requests->back();
      const uint64_t correlation_id = irequest->CorrelationId();

      // If the last queue entry is not an END request then the entire
      // sequence is not contained in the backlog. In that case must
      // update backlog and batcherseqslot maps so that future
      // requests get directed to the batcher sequence-slot instead of
      // the backlog.
      const bool seq_end =
          ((irequest->Flags() & TRITONSERVER_REQUEST_FLAG_SEQUENCE_END) != 0);
      if (!seq_end) {
        // Since the correlation ID is being actively collected in the
        // backlog, there should not be any in-flight sequences with
        // that same correlation ID that have an assigned slot.
        if (sequence_to_batcherseqslot_map_.find(correlation_id) !=
            sequence_to_batcherseqslot_map_.end()) {
          LOG_ERROR << "internal: backlog sequence " << correlation_id
                    << " conflicts with in-flight sequence for model '"
                    << irequest->ModelName() << "'";
        }

        sequence_to_backlog_map_.erase(correlation_id);
        sequence_to_batcherseqslot_map_[correlation_id] = batcher_seq_slot;
      }

      LOG_VERBOSE(1) << "CORRID " << correlation_id << " reusing batcher "
                     << batcher_seq_slot.batcher_idx_ << ", slot "
                     << batcher_seq_slot.seq_slot_ << ": "
                     << irequest->ModelName();
      return correlation_id;
    }
  }

  // There is no backlogged sequence so just release the batch slot
  LOG_VERBOSE(1) << "Freeing slot in batcher " << batcher_seq_slot.batcher_idx_
                 << ", slot " << batcher_seq_slot.seq_slot_;

  ready_batcher_seq_slots_.push(batcher_seq_slot);
  return 0;
}

bool
SequenceBatchScheduler::DelayScheduler(
    const uint32_t batcher_idx, const size_t cnt, const size_t total)
{
  std::unique_lock<std::mutex> lock(mu_);
  queue_request_cnts_[batcher_idx] = cnt;

  size_t seen = 0;
  for (auto c : queue_request_cnts_) {
    seen += c;
  }

  if (seen < total) {
    return true;
  }

  if (backlog_delay_cnt_ > 0) {
    size_t backlog_seen = 0;
    for (const auto& q : backlog_queues_) {
      backlog_seen += q->size();
    }

    if (backlog_seen < backlog_delay_cnt_) {
      return true;
    }
  }

  return false;
}

void
SequenceBatchScheduler::ReaperThread(const int nice)
{
#ifndef _WIN32
  if (setpriority(PRIO_PROCESS, syscall(SYS_gettid), nice) == 0) {
    LOG_VERBOSE(1) << "Starting sequence-batch reaper thread at nice " << nice
                   << "...";
  } else {
    LOG_VERBOSE(1) << "Starting sequence-batch reaper thread at default nice "
                      "(requested nice "
                   << nice << " failed)...";
  }
#else
  LOG_VERBOSE(1) << "Starting sequence-batch reaper thread at default nice...";
#endif

  const uint64_t backlog_idle_wait_microseconds = 50 * 1000;

  while (!reaper_thread_exit_) {
    uint64_t wait_microseconds = max_sequence_idle_microseconds_;
    BatcherSequenceSlotMap force_end_sequences;

    {
      std::unique_lock<std::mutex> lock(mu_);

      uint64_t now_us = std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::steady_clock::now().time_since_epoch())
                            .count();

      for (auto cid_itr = correlation_id_timestamps_.cbegin();
           cid_itr != correlation_id_timestamps_.cend();) {
        int64_t remaining_microseconds =
            (int64_t)max_sequence_idle_microseconds_ -
            (now_us - cid_itr->second);
        if (remaining_microseconds > 0) {
          wait_microseconds =
              std::min(wait_microseconds, (uint64_t)remaining_microseconds + 1);
          ++cid_itr;
          continue;
        }

        const uint64_t idle_correlation_id = cid_itr->first;
        LOG_VERBOSE(1) << "Reaper: CORRID " << idle_correlation_id
                       << ": max sequence idle exceeded";

        auto idle_sb_itr =
            sequence_to_batcherseqslot_map_.find(idle_correlation_id);

        // If the idle correlation ID has an assigned sequence slot,
        // then release that assignment so it becomes available for
        // another sequence. Release is done by enqueuing and must be
        // done outside the lock, so just collect needed info here.
        if (idle_sb_itr != sequence_to_batcherseqslot_map_.end()) {
          force_end_sequences[idle_correlation_id] = idle_sb_itr->second;

          sequence_to_batcherseqslot_map_.erase(idle_correlation_id);
          cid_itr = correlation_id_timestamps_.erase(cid_itr);
        } else {
          // If the idle correlation ID is in the backlog, then just
          // need to increase the timeout so that we revisit it again in
          // the future to check if it is assigned to a sequence slot.
          auto idle_bl_itr = sequence_to_backlog_map_.find(idle_correlation_id);
          if (idle_bl_itr != sequence_to_backlog_map_.end()) {
            LOG_VERBOSE(1) << "Reaper: found idle CORRID "
                           << idle_correlation_id;
            wait_microseconds =
                std::min(wait_microseconds, backlog_idle_wait_microseconds);
            ++cid_itr;
          } else {
            LOG_VERBOSE(1) << "Reaper: ignoring stale idle CORRID "
                           << idle_correlation_id;
            cid_itr = correlation_id_timestamps_.erase(cid_itr);
          }
        }
      }
    }

    // Enqueue force-ends outside of the lock.
    for (const auto& pr : force_end_sequences) {
      const uint64_t idle_correlation_id = pr.first;
      const size_t batcher_idx = pr.second.batcher_idx_;
      const uint32_t seq_slot = pr.second.seq_slot_;

      LOG_VERBOSE(1) << "Reaper: force-ending CORRID " << idle_correlation_id
                     << " in batcher " << batcher_idx << ", slot " << seq_slot;

      // A slot assignment is released by enqueuing a request with a
      // null request. The scheduler thread will interpret the null
      // request as meaning it should release the sequence slot but
      // otherwise do nothing with the request.
      std::unique_ptr<InferenceRequest> null_request;
      batchers_[batcher_idx]->Enqueue(
          seq_slot, idle_correlation_id, null_request);
    }

    // Wait until the next idle timeout needs to be checked
    if (wait_microseconds > 0) {
      std::unique_lock<std::mutex> lock(mu_);
      LOG_VERBOSE(2) << "Reaper: sleeping for " << wait_microseconds << "us...";
      std::chrono::microseconds wait_timeout(wait_microseconds);
      reaper_cv_.wait_for(lock, wait_timeout);
    }
  }

  LOG_VERBOSE(1) << "Stopping sequence-batch reaper thread...";
}

SequenceBatch::SequenceBatch(
    SequenceBatchScheduler* base, const uint32_t batcher_idx,
    const size_t seq_slot_cnt,
    const std::unordered_map<std::string, bool>& enforce_equal_shape_tensors,
    const std::shared_ptr<SequenceBatchScheduler::ControlInputs>&
        start_input_overrides,
    const std::shared_ptr<SequenceBatchScheduler::ControlInputs>&
        end_input_overrides,
    const std::shared_ptr<SequenceBatchScheduler::ControlInputs>&
        startend_input_overrides,
    const std::shared_ptr<SequenceBatchScheduler::ControlInputs>&
        continue_input_overrides,
    const std::shared_ptr<SequenceBatchScheduler::ControlInputs>&
        notready_input_overrides)
    : base_(base), batcher_idx_(batcher_idx), seq_slot_cnt_(seq_slot_cnt),
      enforce_equal_shape_tensors_(enforce_equal_shape_tensors),
      start_input_overrides_(start_input_overrides),
      end_input_overrides_(end_input_overrides),
      startend_input_overrides_(startend_input_overrides),
      continue_input_overrides_(continue_input_overrides),
      notready_input_overrides_(notready_input_overrides)
{
}

bool
SequenceBatch::CreateCorrelationIDControl(const inference::ModelConfig& config)
{
  // If model wants CORRID control then get the name of the input
  // tensor and initialize the override structure for each sequence
  // slot that is used to communicate the correlation ID.
  std::string correlation_id_tensor_name;
  inference::DataType correlation_id_datatype;
  Status corrid_status = GetTypedSequenceControlProperties(
      config.sequence_batching(), config.name(),
      inference::ModelSequenceBatching::Control::CONTROL_SEQUENCE_CORRID,
      false /* required */, &correlation_id_tensor_name,
      &correlation_id_datatype);
  if (!corrid_status.IsOk()) {
    LOG_ERROR << "failed validating CORRID control for sequence-batch "
                 "scheduler thread "
              << batcher_idx_ << ": " << corrid_status.Message();
    return false;
  }

  if (!correlation_id_tensor_name.empty()) {
    if ((correlation_id_datatype != inference::DataType::TYPE_UINT64) &&
        (correlation_id_datatype != inference::DataType::TYPE_INT64) &&
        (correlation_id_datatype != inference::DataType::TYPE_UINT32) &&
        (correlation_id_datatype != inference::DataType::TYPE_INT32)) {
      LOG_ERROR << "unexpected control data type, expected TYPE_UINT64, "
                   "TYPE_INT64, TYPE_UINT32 or TYPE_INT32 for "
                << inference::ModelSequenceBatching_Control_Kind_Name(
                       inference::ModelSequenceBatching::Control::
                           CONTROL_SEQUENCE_CORRID)
                << " for " << config.name();
      return false;
    }

    const std::vector<int64_t> tensor_shape{1};
    std::vector<int64_t> tensor_shape_with_batch_dim{1};
    if (config.max_batch_size() != 0) {
      tensor_shape_with_batch_dim.push_back(1);
    }
    const size_t size_p = GetDataTypeByteSize(correlation_id_datatype);

    for (size_t b = 0; b < seq_slot_cnt_; ++b) {
      TRITONSERVER_MemoryType memory_type;
      int64_t memory_type_id;

      auto corrid_p =
          std::make_shared<AllocatedMemory>(size_p, TRITONSERVER_MEMORY_CPU, 0);
      char* corrid_p_ptr =
          corrid_p->MutableBuffer(&memory_type, &memory_type_id);
      if ((corrid_p_ptr == nullptr) ||
          ((memory_type != TRITONSERVER_MEMORY_CPU) &&
           (memory_type != TRITONSERVER_MEMORY_CPU_PINNED)) ||
          (memory_type_id != 0)) {
        LOG_ERROR << "failed to allocate sequence CORRID control signal in CPU "
                     "memory";
        return false;
      }

      auto override = std::make_shared<InferenceRequest::Input>(
          correlation_id_tensor_name, correlation_id_datatype, tensor_shape);
      *override->MutableShape() = override->OriginalShape();
      *override->MutableShapeWithBatchDim() = tensor_shape_with_batch_dim;
      corrid_status = override->SetData(corrid_p);
      if (!corrid_status.IsOk()) {
        LOG_ERROR << "failed creating CORRID control for sequence-batch "
                     "scheduler thread "
                  << batcher_idx_ << " for " << config.name();
        return false;
      }

      seq_slot_corrid_overrides_.push_back(std::move(override));
    }
  }

  return true;
}

void
SequenceBatch::SetControlTensors(
    std::unique_ptr<InferenceRequest>& irequest, const int32_t seq_slot,
    const uint64_t corrid, const bool not_ready)
{
  const SequenceBatchScheduler::ControlInputs* controls;

  // Set the start, end, and ready control tensors appropriately...
  if (not_ready) {
    controls = notready_input_overrides_.get();
  } else if (
      (irequest->Flags() & (TRITONSERVER_REQUEST_FLAG_SEQUENCE_START |
                            TRITONSERVER_REQUEST_FLAG_SEQUENCE_END)) ==
      (TRITONSERVER_REQUEST_FLAG_SEQUENCE_START |
       TRITONSERVER_REQUEST_FLAG_SEQUENCE_END)) {
    controls = startend_input_overrides_.get();
  } else if (
      (irequest->Flags() & TRITONSERVER_REQUEST_FLAG_SEQUENCE_START) != 0) {
    controls = start_input_overrides_.get();
  } else if (
      (irequest->Flags() & TRITONSERVER_REQUEST_FLAG_SEQUENCE_END) != 0) {
    controls = end_input_overrides_.get();
  } else {
    controls = continue_input_overrides_.get();
  }

  for (const auto& control : *controls) {
    irequest->AddOverrideInput(control);
  }

  // Set correlation ID control tensor if requested by the model.
  if (!seq_slot_corrid_overrides_.empty()) {
    const std::shared_ptr<InferenceRequest::Input>& input =
        seq_slot_corrid_overrides_[seq_slot];
    AllocatedMemory* data =
        reinterpret_cast<AllocatedMemory*>(input->Data().get());
    const char* corrid_p = reinterpret_cast<const char*>(&corrid);
    char* slot_corrid_ptr = data->MutableBuffer();
    memcpy(slot_corrid_ptr, corrid_p, data->TotalByteSize());

    irequest->AddOverrideInput(input);
  }
}

DirectSequenceBatch::DirectSequenceBatch(
    SequenceBatchScheduler* base, const uint32_t batcher_idx,
    const size_t seq_slot_cnt, TritonModelInstance* model_instance,
    const std::unordered_map<std::string, bool>& enforce_equal_shape_tensors,
    const std::shared_ptr<SequenceBatchScheduler::ControlInputs>&
        start_input_overrides,
    const std::shared_ptr<SequenceBatchScheduler::ControlInputs>&
        end_input_overrides,
    const std::shared_ptr<SequenceBatchScheduler::ControlInputs>&
        startend_input_overrides,
    const std::shared_ptr<SequenceBatchScheduler::ControlInputs>&
        continue_input_overrides,
    const std::shared_ptr<SequenceBatchScheduler::ControlInputs>&
        notready_input_overrides,
    bool* is_initialized)
    : SequenceBatch(
          base, batcher_idx, seq_slot_cnt, enforce_equal_shape_tensors,
          start_input_overrides, end_input_overrides, startend_input_overrides,
          continue_input_overrides, notready_input_overrides),
      model_instance_(model_instance), scheduler_thread_exit_(false),
      scheduler_idle_(false), queues_(seq_slot_cnt),
      seq_slot_correlation_ids_(seq_slot_cnt, 0), max_active_seq_slot_(-1)
{
  // Initialize to handle CORRID control. If error just exit
  // now... that means the corresponding model instance will not have
  // any runner and so will not get used for execution.
  const auto& config = model_instance_->Model()->Config();
  if (!CreateCorrelationIDControl(config)) {
    *is_initialized = false;
    return;
  }

  max_batch_size_ = ((size_t)std::max(1, config.max_batch_size()));
  minimum_slot_utilization_ =
      config.sequence_batching().direct().minimum_slot_utilization();
  pending_batch_delay_ns_ =
      config.sequence_batching().direct().max_queue_delay_microseconds() * 1000;

  // Create a scheduler thread associated with 'batcher_idx' that
  // executes the queued requests.
  const int nice = 0;
  scheduler_thread_.reset(
      new std::thread([this, nice]() { BatcherThread(nice); }));

  *is_initialized = true;
}

DirectSequenceBatch::~DirectSequenceBatch()
{
  // Signal the scheduler thread to exit...
  {
    std::unique_lock<std::mutex> lock(mu_);
    scheduler_thread_exit_ = true;
  }

  cv_.notify_one();

  // It is possible for the scheduler thread to be the last holder of
  // a backend object, and when that scheduler thread releases the
  // object the scheduler thread itself will destroy this
  // SequenceBatch object. So we need to check to make sure the
  // scheduler thread does not join it against itself and instead
  // detach it so there is not a problem when its thread object is
  // destroyed.
  if (scheduler_thread_->joinable()) {
    scheduler_thread_->join();
  }
}

void
DirectSequenceBatch::Enqueue(
    const uint32_t seq_slot, const uint64_t correlation_id,
    std::unique_ptr<InferenceRequest>& request)
{
  bool wake_runner = false;

  {
    std::lock_guard<std::mutex> lock(mu_);

    queues_[seq_slot].emplace_back(std::move(request));

    seq_slot_correlation_ids_[seq_slot] = correlation_id;
    max_active_seq_slot_ =
        std::max(max_active_seq_slot_, static_cast<int32_t>(seq_slot));

    // If runner is idle then wake it to service this request. We do
    // the actual wake outside of the lock to avoid having the woken
    // thread immediately block on the lock
    wake_runner = scheduler_idle_;
  }

  if (wake_runner) {
    cv_.notify_one();
  }
}

void
DirectSequenceBatch::NewPayload()
{
  curr_payload_ =
      model_instance_->Model()->Server()->GetRateLimiter()->GetPayload(
          RateLimiter::Payload::Operation::INFER_RUN, model_instance_);
}

void
DirectSequenceBatch::BatcherThread(const int nice)
{
#ifndef _WIN32
  if (setpriority(PRIO_PROCESS, syscall(SYS_gettid), nice) == 0) {
    LOG_VERBOSE(1) << "Starting Direct sequence-batch scheduler thread "
                   << batcher_idx_ << " at nice " << nice << "...";
  } else {
    LOG_VERBOSE(1) << "Starting Direct sequence-batch scheduler thread "
                   << batcher_idx_ << " at default nice (requested nice "
                   << nice << " failed)...";
  }
#else
  LOG_VERBOSE(1) << "Starting Direct sequence-batch scheduler thread "
                 << batcher_idx_ << " at default nice...";
#endif

  // For debugging and testing, delay start of thread until queues
  // contain the specified number of entries (across all
  // SequenceBatchs in the scheduler).
  const char* dstr = getenv("TRITONSERVER_DELAY_SCHEDULER");
  size_t delay_cnt = 0;
  if (dstr != nullptr) {
    delay_cnt = atoi(dstr);
    LOG_VERBOSE(1) << "Delaying scheduler thread " << batcher_idx_ << " until "
                   << delay_cnt << " queued requests...";
  }

  const uint64_t default_wait_microseconds = 500 * 1000;

  NewPayload();

  while (!scheduler_thread_exit_) {
    uint64_t wait_microseconds = default_wait_microseconds;

    // Hold the lock for as short a time as possible.
    {
      std::unique_lock<std::mutex> lock(mu_);

      if (delay_cnt > 0) {
        wait_microseconds = 10 * 1000;
        // Debugging/testing... wait until queues together contain at
        // least 'delay_cnt' items...
        size_t total_size = 0;
        for (const auto& q : queues_) {
          total_size += q.size();
        }
        if (!base_->DelayScheduler(batcher_idx_, total_size, delay_cnt)) {
          delay_cnt = 0;
        }
        LOG_VERBOSE(1) << "Delaying scheduler thread " << batcher_idx_
                       << " until " << delay_cnt
                       << " queued requests, current total = " << total_size;
      } else {
        RequiredEqualInputs required_equal_inputs;
        InferenceRequest* null_irequest = nullptr;

        // Make one pass through the active slots to:
        //
        //   1) release any slots that have forcibly ended sequences
        //
        //   2) find a representative request that will provide:
        //
        //      a) the shape, type, etc. information for null requests
        //
        //      b) the required tensor shapes for the batch for the
        //      case where ragged batching is not allowed
        //
        //   3) Determine the earliest enqueue time and number of ready
        //      sequences if queue delay is enabled
        //
        int32_t max_seq_slot = -1;
        uint64_t earliest_enqueue_time_ns = UINT64_MAX;
        size_t ready_cnt = 0;
        for (int32_t seq_slot = 0; seq_slot <= max_active_seq_slot_;
             ++seq_slot) {
          std::deque<std::unique_ptr<InferenceRequest>>& queue =
              queues_[seq_slot];
          if (!queue.empty()) {
            // If the request is nullptr then the sequence in the slot
            // has timed-out so release the slot for another sequence
            // from the backlog.
            if (queue.front() == nullptr) {
              queue.pop_front();

              SequenceBatchScheduler::BatcherSequenceSlot batcher_seq_slot(
                  batcher_idx_, seq_slot);
              seq_slot_correlation_ids_[seq_slot] =
                  base_->ReleaseSequenceSlot(batcher_seq_slot, &queue);
            }
          }

          // Need to check queue again for contents since if released
          // above it may now be empty...
          if (!queue.empty()) {
            // For NULL requests need an InferenceRequest that can be
            // batched but has controls set to "not ready". Any
            // request can serve this purpose so grab a copy of the
            // first one. This first request is also used to
            // initialize 'required_equal_inputs' so we are sure that
            // this null request will have the correct shape for any
            // created batch.
            if (null_irequest == nullptr) {
              null_irequest = queue.front().get();
            }

            // If this is the first non-null request capture the shape
            // of the tensors that don't support ragged so we can
            // compare them to later requests.
            if (required_equal_inputs.empty() &&
                !enforce_equal_shape_tensors_.empty()) {
              Status status = InitRequiredEqualInputs(
                  queue.front(), enforce_equal_shape_tensors_,
                  &required_equal_inputs);
              if (!status.IsOk()) {
                LOG_ERROR
                    << "internal: unexpecting failure initializing shape: "
                    << status.Message();
                required_equal_inputs.clear();
              }
            }

            earliest_enqueue_time_ns = std::min(
                earliest_enqueue_time_ns, queue.front()->QueueStartNs());
            ready_cnt++;
            max_seq_slot = seq_slot;
          }
        }

        if (max_seq_slot != -1) {
          if ((pending_batch_delay_ns_ == 0) ||
              (minimum_slot_utilization_ == 0.0)) {
            wait_microseconds = 0;
          } else {
            // Compare the age of the oldest pending request to the maximum
            // batch queuing delay, and the size of the ready requests in the
            // batch, execute now if queuing delay is exceeded or the batch size
            // is large enough. Otherwise create a timer to wakeup a thread to
            // check again at the maximum allowed delay.
            uint64_t now_ns =
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch())
                    .count();
            uint64_t current_batch_delay_ns =
                (now_ns - earliest_enqueue_time_ns);
            if ((current_batch_delay_ns > pending_batch_delay_ns_) ||
                (((float)ready_cnt) / max_batch_size_ >=
                 minimum_slot_utilization_)) {
              wait_microseconds = 0;
              LOG_VERBOSE(1)
                  << "start sequence batch execution. "
                  << "current batch delay: " << current_batch_delay_ns
                  << "; maximum delay allowed: " << pending_batch_delay_ns_
                  << "slot utilization: " << ready_cnt << "/" << max_batch_size_
                  << "; utilization threshold: " << minimum_slot_utilization_;
            } else {
              wait_microseconds =
                  (pending_batch_delay_ns_ - current_batch_delay_ns) / 1000;
              // reset 'max_seq_slot' so that not request is pulled from the
              // queues
              max_seq_slot = -1;
              LOG_VERBOSE(1)
                  << "defer sequence batch execution. "
                  << "current batch delay: " << current_batch_delay_ns
                  << "; maximum delay allowed: " << pending_batch_delay_ns_
                  << "slot utilization: " << ready_cnt << "/" << max_batch_size_
                  << "; utilization threshold: " << minimum_slot_utilization_;
            }
          }
        }

        // Collect requests from slot 0 to max_seq_slot.
        for (int32_t seq_slot = 0; seq_slot <= max_seq_slot; ++seq_slot) {
          bool end_of_sequence = false;
          bool use_null_request = false;
          std::deque<std::unique_ptr<InferenceRequest>>& queue =
              queues_[seq_slot];

          // If 'seq_slot' doesn't have any requests then change the
          // request to send dummy/null input tensors for this
          // slot. We need this so that other requests stay in the
          // correct slot.
          if (queue.empty()) {
            use_null_request = true;
          }
          // If there are one or more tensors that don't support
          // ragged batch, then don't allow a request into an existing
          // batch if shape differs.
          else if (
              !required_equal_inputs.empty() &&
              !enforce_equal_shape_tensors_.empty()) {
            if (!CompareWithRequiredEqualInputs(
                    queue.front(), required_equal_inputs)) {
              use_null_request = true;
            }
          }

          // Use null-request if necessary otherwise use the next
          // request in the queue...
          if (use_null_request) {
            std::unique_ptr<InferenceRequest> ni(
                InferenceRequest::CopyAsNull(*null_irequest));
            // Note that when the not-ready control input of the
            // request is "true" the model can't assume that any
            // other inputs are meaningful, including CORRID. So we
            // just use zero for that.
            SetControlTensors(
                ni, seq_slot, 0 /* corrid */, true /* not_ready */);
            curr_payload_->AddRequest(std::move(ni));
          } else {
            std::unique_ptr<InferenceRequest>& irequest = queue.front();

            // Set the control tensor values in the request.
            SetControlTensors(
                irequest, seq_slot, seq_slot_correlation_ids_[seq_slot]);

            if ((irequest->Flags() & TRITONSERVER_REQUEST_FLAG_SEQUENCE_END) !=
                0) {
              end_of_sequence = true;
            }

            curr_payload_->AddRequest(std::move(irequest));

            queue.pop_front();
          }

          if (curr_payload_->GetState() ==
              RateLimiter::Payload::State::UNINITIALIZED) {
            curr_payload_->SetState(RateLimiter::Payload::State::READY);
          }

          // If the sequence has ended then attempt to refill the
          // sequence slot with a sequence from the backlog. If
          // there is no backlog show that the slot is no longer
          // active.
          if (end_of_sequence) {
            LOG_VERBOSE(1) << "End sequence CORRID "
                           << seq_slot_correlation_ids_[seq_slot]
                           << " in batcher " << batcher_idx_ << ", slot "
                           << seq_slot;

            // Should never be anything in a queue after the END
            // marker. If it happens that means we will clobber
            // that request if/when we swap in a backlog sequence
            // in ReleaseSequenceSlot below.
            if (!queue.empty()) {
              LOG_ERROR << "internal: unexpected requests after sequence "
                           "end in slot "
                        << seq_slot;
            }

            SequenceBatchScheduler::BatcherSequenceSlot batcher_seq_slot(
                batcher_idx_, seq_slot);
            seq_slot_correlation_ids_[seq_slot] =
                base_->ReleaseSequenceSlot(batcher_seq_slot, &queue);
          }
        }
      }

      // One or more sequences may have ended... find the new
      // 'max_active_seq_slot_'.
      while ((max_active_seq_slot_ >= 0) &&
             (seq_slot_correlation_ids_[max_active_seq_slot_] == 0)) {
        max_active_seq_slot_--;
      }

      // If no requests are to be handled, wait for notification or
      // for the specified timeout before checking the queues again.
      if (wait_microseconds > 0) {
        scheduler_idle_ = true;
        std::chrono::microseconds wait_timeout(wait_microseconds);
        cv_.wait_for(lock, wait_timeout);
        scheduler_idle_ = false;
      }
    }

    if (curr_payload_->GetState() == RateLimiter::Payload::State::READY) {
      // Run the backend...
      model_instance_->Model()->Server()->GetRateLimiter()->EnqueuePayload(
          model_instance_->Model(), curr_payload_);
      NewPayload();
    }
  }  // end runner loop

  LOG_VERBOSE(1) << "Stopping Direct sequence-batch scheduler thread "
                 << batcher_idx_ << "...";
}

OldestSequenceBatch::OldestSequenceBatch(
    SequenceBatchScheduler* base, const uint32_t batcher_idx,
    const size_t seq_slot_cnt, TritonModelInstance* model_instance,
    const std::unordered_map<std::string, bool>& enforce_equal_shape_tensors,
    const std::shared_ptr<SequenceBatchScheduler::ControlInputs>&
        start_input_overrides,
    const std::shared_ptr<SequenceBatchScheduler::ControlInputs>&
        end_input_overrides,
    const std::shared_ptr<SequenceBatchScheduler::ControlInputs>&
        startend_input_overrides,
    const std::shared_ptr<SequenceBatchScheduler::ControlInputs>&
        continue_input_overrides,
    const std::shared_ptr<SequenceBatchScheduler::ControlInputs>&
        notready_input_overrides,
    bool* is_initialized)
    : SequenceBatch(
          base, batcher_idx, seq_slot_cnt, enforce_equal_shape_tensors,
          start_input_overrides, end_input_overrides, startend_input_overrides,
          continue_input_overrides, notready_input_overrides),
      in_flight_(seq_slot_cnt, false), queues_(seq_slot_cnt)
{
  // Initialize to handle CORRID control. If error just exit
  // now... that means the corresponding model instance will not have
  // any runner and so will not get used for execution.
  const auto& config = model_instance->Model()->Config();
  if (!CreateCorrelationIDControl(config)) {
    *is_initialized = false;
    return;
  }

  // Create a dynamic batcher use to batch together sequences for
  // inference.
  std::set<int32_t> preferred_batch_sizes;
  for (const auto size :
       config.sequence_batching().oldest().preferred_batch_size()) {
    preferred_batch_sizes.insert(size);
  }

  Status status = DynamicBatchScheduler::Create(
      model_instance->Model(), model_instance, GetCpuNiceLevel(config),
      true /* dynamic_batching_enabled */, config.max_batch_size(),
      enforce_equal_shape_tensors_, true /* preserve_ordering */,
      preferred_batch_sizes,
      config.sequence_batching().oldest().max_queue_delay_microseconds(),
      &dynamic_batcher_);
  if (!status.IsOk()) {
    LOG_ERROR << "failed creating dynamic sequence batcher for OldestFirst "
              << batcher_idx_ << ": " << status.Message();
    *is_initialized = false;
    return;
  }

  *is_initialized = true;
}
OldestSequenceBatch::~OldestSequenceBatch() {}

void
OldestSequenceBatch::CompleteAndNext(const uint32_t seq_slot)
{
  std::lock_guard<std::mutex> lock(mu_);

  // We may enqueue 1 or more pending inferences triggered by the
  // completion. If the sequence has a pending inference then it needs
  // to be send to dynamic batcher since the "previous" inference just
  // completed. If this next inference ends up being the end of the
  // sequence (either from the END flag or because the sequence is
  // being force-ended) then we try to fill the now-free sequence slot
  // from the backlog and then send the first inference from that
  // sequence to the dynamic batcher...
  std::deque<std::unique_ptr<InferenceRequest>>& queue = queues_[seq_slot];
  bool retry = true;
  while (retry) {
    retry = false;

    bool release_seq_slot = false;
    in_flight_[seq_slot] = false;

    // If the next sequence inference is ready in the queue then enqueue
    // it in the dynamic batcher now.
    if (!queue.empty()) {
      auto& irequest = queue.front();

      // If the request is null then this inference request is from
      // the reaper thread indicating a timed-out sequence. Mark that
      // the sequence slot should be released but otherwise do
      // nothing.
      if (irequest == nullptr) {
        LOG_VERBOSE(1) << "force-end sequence in batcher " << batcher_idx_
                       << ", slot " << seq_slot;
        release_seq_slot = true;
      } else {
        const uint64_t correlation_id = irequest->CorrelationId();

        // After handling the last inference in a sequence we must
        // release the sequence slot to make it available to another
        // sequence.
        if ((irequest->Flags() & TRITONSERVER_REQUEST_FLAG_SEQUENCE_END) != 0) {
          LOG_VERBOSE(1) << "end sequence CORRID " << correlation_id
                         << " in batcher " << batcher_idx_ << ", slot "
                         << seq_slot;
          release_seq_slot = true;
        }

        // Add the appropriate control tensor values to the request.
        SetControlTensors(irequest, seq_slot, correlation_id);

        LOG_VERBOSE(1) << "issue to dynamic batcher CORRID " << correlation_id
                       << " in batcher " << batcher_idx_ << ", slot "
                       << seq_slot;
        in_flight_[seq_slot] = true;

        irequest->AddInternalReleaseCallback(
            [this, seq_slot]() { CompleteAndNext(seq_slot); });

        dynamic_batcher_->Enqueue(irequest);
      }

      queue.pop_front();
    }

    // If releasing the sequence slot then the sequence queue should be
    // empty and we can now assign a new sequence to the queue (from the
    // backlog).
    if (release_seq_slot) {
      // Should never be anything in a queue after the END marker. If it
      // happens that means we will clobber that request if/when we swap
      // in a backlog sequence in ReleaseSequenceSlot below.
      if (!queue.empty()) {
        LOG_ERROR << "internal: unexpected requests after sequence end in slot "
                  << seq_slot;
      }

      SequenceBatchScheduler::BatcherSequenceSlot batcher_seq_slot(
          batcher_idx_, seq_slot);
      const uint64_t released_cid =
          base_->ReleaseSequenceSlot(batcher_seq_slot, &queue);
      if (released_cid != 0) {
        LOG_VERBOSE(1) << "Enqueued new sequence containing " << queue.size()
                       << " requests into OldestFirst batcher " << batcher_idx_
                       << ", slot " << seq_slot;

        // If an inference is already in-flight in the dynamic batcher
        // in this sequence slot then can't process the new queue
        // inferences right now, because the in-flight request is
        // using slot resources like the CORRID override map.
        if (!in_flight_[seq_slot]) {
          retry = true;
        }
      }
    }
  }
}

void
OldestSequenceBatch::Enqueue(
    const uint32_t seq_slot, const uint64_t correlation_id,
    std::unique_ptr<InferenceRequest>& request)
{
  // Queue the new request... if there isn't already a request in
  // flight for this sequence then send one to the dynamic batcher
  // immediately.
  bool in_flight;
  {
    std::lock_guard<std::mutex> lock(mu_);

    std::deque<std::unique_ptr<InferenceRequest>>& queue = queues_[seq_slot];
    queue.emplace_back(std::move(request));
    in_flight = in_flight_[seq_slot];
  }

  if (!in_flight) {
    CompleteAndNext(seq_slot);
  }
}

}}  // namespace nvidia::inferenceserver
