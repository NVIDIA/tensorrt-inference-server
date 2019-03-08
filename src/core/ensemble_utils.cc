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

#include "src/core/ensemble_utils.h"

#include <set>
#include "absl/strings/numbers.h"
#include "src/core/constants.h"
#include "src/core/logging.h"
#include "src/core/model_config_utils.h"

namespace nvidia { namespace inferenceserver {

std::string
DimsListToString(const DimsList& list)
{
  std::string res = "[ ";
  for (const auto& dim : list) {
    res = res + std::to_string(dim) + " ";
  }
  return res + "]";
}

tensorflow::Status
ValidateTensorConsistency(
    const TensorNode& lhs, const TensorNode& rhs, const std::string& message)
{
  if (lhs.type != rhs.type) {
    return tensorflow::errors::InvalidArgument(
        message, "inconsistent data type: ", lhs.type,
        " is inferred from model ", lhs.model_name, " while ", rhs.type,
        " is inferred from model ", rhs.model_name);
  }
  bool consistent = (lhs.dims.size() == rhs.dims.size());
  if (consistent) {
    for (size_t i = 0; i < lhs.dims.size(); i++) {
      if (lhs.dims[i] != rhs.dims[i]) {
        consistent = false;
        break;
      }
    }
  }
  if (!consistent) {
    return tensorflow::errors::InvalidArgument(
        message, "inconsistent shape: ", DimsListToString(lhs.dims),
        " is inferred from model ", lhs.model_name, " while ",
        DimsListToString(rhs.dims), " is inferred from model ", rhs.model_name);
  }
  return tensorflow::Status::OK();
}

tensorflow::Status
ValidateEnsembleConfig(
    const std::string& ensemble,
    const std::unordered_map<std::string, ModelConfig>& config_map,
    const std::unordered_map<std::string, std::string>& invalid_model_names,
    std::unordered_map<std::string, bool>& ensembles,
    std::deque<std::string>& ensemble_dependency)
{
  std::unordered_map<std::string, TensorNode> ensemble_tensors;

  const auto& ensemble_config = config_map.at(ensemble);

  for (const auto& input : ensemble_config.input()) {
    TensorNode input_node(ensemble, input.data_type(), input.dims());
    ensemble_tensors.emplace(std::make_pair(input.name(), input_node));
  }
  for (const auto& output : ensemble_config.output()) {
    TensorNode output_node(ensemble, output.data_type(), output.dims());
    ensemble_tensors.emplace(std::make_pair(output.name(), output_node));
  }

  for (const auto& step : ensemble_config.ensemble_scheduling().step()) {
    const auto& model_name = step.model_name();
    if (invalid_model_names.find(model_name) != invalid_model_names.end()) {
      return tensorflow::errors::InvalidArgument(
          "ensemble ", ensemble, " contains invalid model ", model_name, " : ",
          invalid_model_names.at(model_name));
    }
    auto it = config_map.find(model_name);
    if (it == config_map.end()) {
      return tensorflow::errors::InvalidArgument(
          "ensemble ", ensemble, " contains model ", model_name,
          " which is not in the available models");
    }
    const auto& model_config = it->second;
    if (model_config.max_batch_size() < ensemble_config.max_batch_size()) {
      return tensorflow::errors::InvalidArgument(
          "ensemble ", ensemble, " allows maximum batch size ",
          ensemble_config.max_batch_size(), ", but it contains model ",
          model_name, " which only allows  maximum batch size to be ",
          model_config.max_batch_size());
    }

    if (model_config.has_ensemble_scheduling()) {
      for (const auto& name : ensemble_dependency) {
        if (name == model_name) {
          return tensorflow::errors::InvalidArgument(
              "circular dependency between ensembles: ", name, " -> ... -> ",
              ensemble, " -> ", name);
        }
      }

      if ((ensembles.find(model_name))->second == false) {
        ensemble_dependency.push_back(ensemble);
        TF_RETURN_IF_ERROR(ValidateEnsembleConfig(
            model_name, config_map, invalid_model_names, ensembles,
            ensemble_dependency));
        ensemble_dependency.pop_back();
      }
    }

    // Check all inputs are mapped and no mapping to invalid inputs
    std::set<std::string> input_names;
    for (const auto& model_input : model_config.input()) {
      input_names.insert(model_input.name());
    }
    for (const auto& input_map : step.input_map()) {
      if (input_names.find(input_map.second) == input_names.end()) {
        return tensorflow::errors::InvalidArgument(
            "in ensemble ", ensemble, ", ensemble tensor ", input_map.first,
            " is mapping to non-existing input ", input_map.second,
            " in model ", step.model_name());
      }
    }
    for (const auto& model_input : model_config.input()) {
      bool found = false;
      for (const auto& input_map : step.input_map()) {
        found = (model_input.name() == input_map.second);
        if (found) {
          TensorNode model_tensor(
              step.model_name(), model_input.data_type(), model_input.dims());
          auto it = ensemble_tensors.find(input_map.first);
          if (it != ensemble_tensors.end()) {
            TF_RETURN_IF_ERROR(ValidateTensorConsistency(
                it->second, model_tensor,
                "in ensemble " + ensemble + ", ensemble tensor " +
                    input_map.first + ": "));
          } else {
            ensemble_tensors.emplace(
                std::make_pair(input_map.first, model_tensor));
          }
        }
      }
      if (!found) {
        return tensorflow::errors::InvalidArgument(
            "in ensemble ", ensemble, ", input ", model_input.name(),
            " in model ", model_config.name(),
            " is not mapped to any ensemble tensors");
      }
    }

    // Check no multiple mappings to same ensemble tensor
    // and no mapping from invalid outputs
    std::set<std::string> mapped;
    for (const auto& output_map : step.output_map()) {
      if (mapped.find(output_map.second) == mapped.end()) {
        mapped.insert(output_map.second);
      } else {
        return tensorflow::errors::InvalidArgument(
            "in ensemble " + ensemble + ", multiple outputs in model ",
            model_config.name(), " are mapped to the same ensemble tensor ",
            output_map.second);
      }
      bool found = false;
      for (const auto& model_output : model_config.output()) {
        found = (model_output.name() == output_map.first);
        if (found) {
          TensorNode model_tensor(
              step.model_name(), model_output.data_type(), model_output.dims());
          auto it = ensemble_tensors.find(output_map.second);
          if (it != ensemble_tensors.end()) {
            TF_RETURN_IF_ERROR(ValidateTensorConsistency(
                it->second, model_tensor,
                "in ensemble " + ensemble + ", ensemble tensor " +
                    output_map.second + ": "));
          } else {
            ensemble_tensors.emplace(
                std::make_pair(output_map.second, model_tensor));
          }
        }
      }
      if (!found) {
        return tensorflow::errors::InvalidArgument(
            "in ensemble ", ensemble, ", ensemble tensor ", output_map.second,
            " is mapped from non-existing output ", output_map.first,
            " in model ", step.model_name());
      }
    }

    // link ensemble tensors
    for (const auto& output_map : step.output_map()) {
      auto& node = ensemble_tensors.find(output_map.second)->second;
      for (const auto& input_map : step.input_map()) {
        auto& prev_node = ensemble_tensors.find(input_map.first)->second;
        node.prev_nodes.push_back(&prev_node);
        prev_node.next_nodes.push_back(&node);
      }
    }
  }

  // Check data flow
  std::deque<TensorNode*> ready_queue;
  for (const auto& input : ensemble_config.input()) {
    auto it = ensemble_tensors.find(input.name());
    it->second.ready = true;
    ready_queue.push_back(&(it->second));
  }
  while (!ready_queue.empty()) {
    auto& ready_node = ready_queue.front();
    for (auto& next_node : ready_node->next_nodes) {
      if (next_node->ready) {
        continue;
      }
      bool next_node_ready = true;
      for (auto& prev_node : next_node->prev_nodes) {
        if (!prev_node->ready) {
          next_node_ready = false;
          break;
        }
      }
      next_node->ready = next_node_ready;
      if (next_node_ready) {
        ready_queue.push_back(next_node);
      }
    }
    ready_queue.pop_front();
  }
  for (const auto& output : ensemble_config.output()) {
    auto it = ensemble_tensors.find(output.name());
    if (!it->second.ready) {
      return tensorflow::errors::InvalidArgument(
          "in ensemble ", ensemble, ", no data will be written to ",
          "ensemble output ", output.name(), " under optimistic assumption");
    }
  }
  (ensembles.find(ensemble))->second = true;
  return tensorflow::Status::OK();
}

tensorflow::Status
ValidateEnsembleConfig(
    const std::unordered_map<std::string, ModelConfig>& config_map)
{
  std::unordered_map<std::string, std::string> invalid_model_names;
  std::unordered_map<std::string, bool> ensembles;

  for (const auto& pair : config_map) {
    tensorflow::Status status;
    for (const auto& input : pair.second.input()) {
      status = ValidateModelInput(input);
      if (!status.ok()) {
        break;
      }
    }
    if (status.ok()) {
      for (const auto& output : pair.second.output()) {
        status = ValidateModelOutput(output);
        if (!status.ok()) {
          break;
        }
      }
    }
    if (!status.ok()) {
      // Return error if the inputs / outputs of one ensemble is not correct.
      if (pair.second.has_ensemble_scheduling()) {
        return tensorflow::errors::InvalidArgument(
            "ensemble", pair.first, ": ", status.error_message());
      }
      invalid_model_names.emplace(pair.first, status.error_message());
    } else if (pair.second.has_ensemble_scheduling()) {
      ensembles.emplace(std::make_pair(pair.first, false));
    }
  }

  std::deque<std::string> ensemble_dependency;
  for (const auto& pair : ensembles) {
    if (pair.second) {
      continue;
    }
    TF_RETURN_IF_ERROR(ValidateEnsembleConfig(
        pair.first, config_map, invalid_model_names, ensembles,
        ensemble_dependency));
  }
  return tensorflow::Status::OK();
}

}}  // namespace nvidia::inferenceserver
