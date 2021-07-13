# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required (VERSION 3.18)

#
# Model configuration utilities used by both clients and server.
#
add_library(
  model-config-library EXCLUDE_FROM_ALL OBJECT
  model_config.cc model_config.h
)
target_link_libraries(
  model-config-library
  PUBLIC
    proto-library           # from repo-common
    triton-core-serverapi   # from repo-core
)

if(${TRITON_ENABLE_GPU})
  add_library(
    model-config-cuda-library EXCLUDE_FROM_ALL OBJECT
    model_config_cuda.cc model_config_cuda.h
  )
  target_include_directories(model-config-cuda-library PRIVATE ${CUDA_INCLUDE_DIRS})
  target_link_libraries(
    model-config-cuda-library
    PUBLIC
      proto-library     # from repo-common
  )
endif() # TRITON_ENABLE_GPU

#
# Triton repo agent API
#
set(
  REPOAGENT_SRCS
  triton_repo_agent.cc
)

set(
  REPOAGENT_HDRS
  triton_repo_agent.h
)

add_library(
  triton-repoagent-library EXCLUDE_FROM_ALL OBJECT
  ${REPOAGENT_SRCS} ${REPOAGENT_HDRS}
)
target_link_libraries(
  triton-repoagent-library
  PRIVATE
    proto-library             # from repo-common
  PUBLIC
    triton-core-repoagentapi  # from repo-core
    triton-core-serverapi     # from repo-core
    triton-common-json        # from repo-common
)

#
# Inference server core
#
if(${TRITON_ENABLE_GCS})
  find_package(storage_client REQUIRED)
  message(STATUS "Using google-cloud-cpp ${storage_client_VERSION}")
if (NOT WIN32)
  set_source_files_properties(filesystem.cc PROPERTIES COMPILE_FLAGS -Wno-missing-field-initializers)
endif() # NOT WIN32
endif() # TRITON_ENABLE_GCS

if(${TRITON_ENABLE_S3})
  find_package(AWSSDK REQUIRED COMPONENTS s3)
  message(STATUS "Using aws-sdk-cpp ${AWSSDK_VERSION}")
endif()

if(${TRITON_ENABLE_GPU})
  set(HOST_COMPILER_FLAGS "")
  if (WIN32)
    set(HOST_COMPILER_FLAGS "/MD")
  else()
    set(HOST_COMPILER_FLAGS "-fPIC")
  endif()

  set(CUDA_LIBRARIES PUBLIC ${CUDA_LIBRARIES})
  cuda_add_library(
    kernel-library
    kernel.cu kernel.h
    OPTIONS -arch compute_53
    OPTIONS -code compute_53,sm_53,sm_60,sm_61,sm_62,sm_70,sm_72,sm_75
    OPTIONS -Xcompiler ${HOST_COMPILER_FLAGS}
  )
endif() # TRITON_ENABLE_GPU

set(
  SERVER_SRCS
  autofill.cc
  backend.cc
  backend_context.cc
  cuda_utils.cc
  dynamic_batch_scheduler.cc
  ensemble_scheduler.cc
  ensemble_utils.cc
  filesystem.cc
  infer_parameter.cc
  infer_request.cc
  infer_response.cc
  label_provider.cc
  logging.cc
  memory.cc
  metric_model_reporter.cc
  metrics.cc
  model_config_utils.cc
  model_repository_manager.cc
  numa_utils.cc
  persistent_backend_manager.cc
  pinned_memory_manager.cc
  scheduler_utils.cc
  sequence_batch_scheduler.cc
  server.cc
  shared_library.cc
  infer_stats.cc
  infer_trace.cc
  rate_limiter.cc
  status.cc
  tritonserver.cc
)

set(
  SERVER_HDRS
  autofill.h
  backend.h
  backend_context.h
  constants.h
  cuda_utils.h
  dynamic_batch_scheduler.h
  ensemble_scheduler.h
  ensemble_utils.h
  filesystem.h
  infer_parameter.h
  infer_request.h
  infer_response.h
  label_provider.h
  logging.h
  memory.h
  metric_model_reporter.h
  metrics.h
  model_config_utils.h
  model_repository_manager.h
  numa_utils.h
  nvtx.h
  persistent_backend_manager.h
  pinned_memory_manager.h
  rate_limiter.h
  response_allocator.h
  scheduler.h
  scheduler_utils.h
  sequence_batch_scheduler.h
  server.h
  server_message.h
  shared_library.h
  infer_stats.h
  infer_trace.h
  status.h
)

if(${TRITON_ENABLE_GPU})
  set(
    SERVER_SRCS
    ${SERVER_SRCS}
    cuda_memory_manager.cc
  )
  set(
    SERVER_HDRS
    ${SERVER_HDRS}
    cuda_memory_manager.h
    kernel.h
  )
endif() # TRITON_ENABLE_GPU

add_library(
  server-library EXCLUDE_FROM_ALL OBJECT
  ${SERVER_SRCS} ${SERVER_HDRS}
)

if(${TRITON_ENABLE_GPU})
  target_include_directories(
    server-library
    PRIVATE ${CUDA_INCLUDE_DIRS}
    PRIVATE ${CNMEM_PATH}/include
  )
endif() # TRITON_ENABLE_GPU

if(${TRITON_ENABLE_METRICS})
  target_include_directories(
    server-library
    PRIVATE $<TARGET_PROPERTY:prometheus-cpp::core,INTERFACE_INCLUDE_DIRECTORIES>
  )
endif() # TRITON_ENABLE_METRICS

if(${TRITON_ENABLE_GCS})
  target_include_directories(
    server-library
    PRIVATE $<TARGET_PROPERTY:storage_client,INTERFACE_INCLUDE_DIRECTORIES>
  )
endif() # TRITON_ENABLE_GCS

if(${TRITON_ENABLE_S3})
  target_include_directories(
    server-library
    PRIVATE $<TARGET_PROPERTY:aws-cpp-sdk-s3,INTERFACE_INCLUDE_DIRECTORIES>
  )
endif() # TRITON_ENABLE_S3

if(${TRITON_ENABLE_AZURE_STORAGE})
  target_include_directories(
    server-library
    PRIVATE ${azure-storage-cpplite_DIR}/include
  )
endif() # TRITON_ENABLE_AZURE_STORAGE

target_link_libraries(
  server-library
  PRIVATE
    proto-library                    # from repo-common
  PUBLIC
    triton-core-backendapi           # from repo-core
    triton-core-serverapi            # from repo-core
    triton-common-async-work-queue   # from repo-common
    triton-common-sync-queue         # from repo-common
    triton-common-json               # from repo-common
    triton-common-table-printer      # from repo-common
)
