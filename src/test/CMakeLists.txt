# Copyright (c) 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# caffe2plan
#
if(${TRITON_ENABLE_TENSORRT})
add_executable(caffe2plan caffe2plan.cc)
target_include_directories(caffe2plan PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(
  caffe2plan
  PRIVATE -lnvinfer
  PRIVATE -lnvcaffe_parser
  PRIVATE ${CUDA_LIBRARIES}
)
install(
  TARGETS caffe2plan
  RUNTIME DESTINATION bin
)
endif() # TRITON_ENABLE_TENSORRT

#
# CudaMemoryManger
#
set(
  CUDA_MEMORY_MANAGER_SRCS
  ../core/cuda_memory_manager.cc
  ../core/status.cc
  ../core/cuda_utils.cc
)

set(
  CUDA_MEMORY_MANAGER_HDRS
  ../core/cuda_memory_manager.h
  ../core/status.h
  ../core/cuda_utils.h
)

#
# PinnedMemoryManger
#
set(
  PINNED_MEMORY_MANAGER_SRCS
  ../core/pinned_memory_manager.cc
  ../core/status.cc
  ../core/cuda_utils.cc
  ../core/numa_utils.cc
)

set(
  PINNED_MEMORY_MANAGER_HDRS
  ../core/pinned_memory_manager.h
  ../core/status.h
  ../core/cuda_utils.h
  ../core/numa_utils.h
)

#
# Memory
#
set(
  MEMORY_SRCS
  ../core/memory.cc
)

set(
  MEMORY_HDRS
  ../core/memory.h
  ../core/constants.h
)

set(
  MEMORY_TEST_SRCS
  memory_test.cc
  ${MEMORY_SRCS}
  ${CUDA_MEMORY_MANAGER_SRCS}
  ${PINNED_MEMORY_MANAGER_SRCS}
)

set(
  MEMORY_TEST_HDRS
  ${MEMORY_HDRS}
  ${CUDA_MEMORY_MANAGER_HDRS}
  ${PINNED_MEMORY_MANAGER_HDRS}
)

find_package(GTest REQUIRED)
add_executable(
  memory_test
  ${MEMORY_TEST_SRCS}
  ${MEMORY_TEST_HDRS}
  $<TARGET_OBJECTS:proto-library>
)
set_target_properties(
  memory_test
  PROPERTIES
    SKIP_BUILD_RPATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH FALSE
    INSTALL_RPATH ""
)
target_include_directories(
  memory_test
  PRIVATE ${GTEST_INCLUDE_DIR}
  PRIVATE ${CUDA_INCLUDE_DIRS}
  PRIVATE ${CNMEM_PATH}/include
)
target_link_libraries(
  memory_test
  PRIVATE triton-common-error        # from repo-common
  PRIVATE triton-common-sync-queue   # from repo-common
  PRIVATE triton-core-serverapi      # from repo-core
  PRIVATE proto-library              # from repo-common
  PRIVATE ${GTEST_LIBRARY}
  PRIVATE ${GTEST_MAIN_LIBRARY}
  PRIVATE ${CUDA_LIBRARIES}
  PRIVATE -lpthread
  PRIVATE protobuf::libprotobuf
  PRIVATE -L${CNMEM_PATH}/lib
  PRIVATE -lcnmem
  PRIVATE numa
)
install(
  TARGETS memory_test
  RUNTIME DESTINATION bin
)

set(
  PINNED_MEMORY_MANAGER_TEST_SRCS
  pinned_memory_manager_test.cc
  ${PINNED_MEMORY_MANAGER_SRCS}
)

set(
  PINNED_MEMORY_MANAGER_TEST_HDRS
  ${PINNED_MEMORY_MANAGER_HDRS}
)

find_package(GTest REQUIRED)
add_executable(
  pinned_memory_manager_test
  ${PINNED_MEMORY_MANAGER_TEST_SRCS}
  ${PINNED_MEMORY_MANAGER_TEST_HDRS}
  $<TARGET_OBJECTS:proto-library>
)
set_target_properties(
  pinned_memory_manager_test
  PROPERTIES
    SKIP_BUILD_RPATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH FALSE
    INSTALL_RPATH ""
)
target_include_directories(
  pinned_memory_manager_test
  PRIVATE ${GTEST_INCLUDE_DIR}
  PRIVATE ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(
  pinned_memory_manager_test
  PRIVATE triton-common-error        # from repo-common
  PRIVATE triton-common-sync-queue   # from repo-common
  PRIVATE triton-core-serverapi      # from repo-core
  PRIVATE proto-library              # from repo-common
  PRIVATE ${GTEST_LIBRARY}
  PRIVATE ${GTEST_MAIN_LIBRARY}
  PRIVATE ${CUDA_LIBRARIES}
  PRIVATE protobuf::libprotobuf
  PRIVATE -lpthread
  PRIVATE numa
)
install(
  TARGETS pinned_memory_manager_test
  RUNTIME DESTINATION bin
)

#
# Unit test for AsyncWorkQueue
#
set(
  ASYNC_WORK_QUEUE_SRCS
  ../core/status.cc
)

set(
  ASYNC_WORK_QUEUE_HDRS
  ../core/status.h
)

set(
  ASYNC_WORK_QUEUE_TEST_SRCS
  async_work_queue_test.cc
  ${ASYNC_WORK_QUEUE_SRCS}
)

set(
  ASYNC_WORK_QUEUE_TEST_HDRS
  ${ASYNC_WORK_QUEUE_HDRS}
)

find_package(GTest REQUIRED)
add_executable(
  async_work_queue_test
  ${ASYNC_WORK_QUEUE_TEST_SRCS}
  ${ASYNC_WORK_QUEUE_TEST_HDRS}
)
set_target_properties(
  async_work_queue_test
  PROPERTIES
    SKIP_BUILD_RPATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH FALSE
    INSTALL_RPATH ""
)
target_include_directories(
  async_work_queue_test
  PRIVATE ${GTEST_INCLUDE_DIR}
)
target_link_libraries(
  async_work_queue_test
  PRIVATE triton-core-serverapi            # from repo-core
  PRIVATE triton-common-sync-queue         # from repo-common
  PRIVATE triton-common-async-work-queue   # from repo-common
  PRIVATE triton-common-error              # from repo-common
  PRIVATE ${GTEST_LIBRARY}
  PRIVATE ${GTEST_MAIN_LIBRARY}
  PRIVATE -lpthread
)
install(
  TARGETS async_work_queue_test
  RUNTIME DESTINATION bin
)

#
# Unit test for DataCompressor
#
set(
  DATA_COMPRESSOR_HDRS
  ../servers/data_compressor.h
  ../servers/common.h
)

set(
  DATA_COMPRESSOR_TEST_SRCS
  data_compressor_test.cc
)

set(
  DATA_COMPRESSOR_TEST_HDRS
  ${DATA_COMPRESSOR_HDRS}
)

find_package(GTest REQUIRED)
find_package(Libevent CONFIG REQUIRED)
include_directories(${LIBEVENT_INCLUDE_DIRS})
add_executable(
  data_compressor_test
  ${DATA_COMPRESSOR_TEST_SRCS}
  ${DATA_COMPRESSOR_TEST_HDRS}
)
set_target_properties(
  data_compressor_test
  PROPERTIES
    SKIP_BUILD_RPATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH FALSE
    INSTALL_RPATH ""
)
target_include_directories(
  data_compressor_test
  PRIVATE ${GTEST_INCLUDE_DIR}
)
target_link_libraries(
  data_compressor_test
  PRIVATE triton-core-serverapi  # from repo-core
  PRIVATE proto-library          # from repo-common
  PRIVATE ${GTEST_LIBRARY}
  PRIVATE ${GTEST_MAIN_LIBRARY}
  PRIVATE ${LIBEVENT_LIBRARIES}
  PRIVATE protobuf::libprotobuf
  PRIVATE -lpthread
  PRIVATE -lz
)
install(
  TARGETS data_compressor_test
  RUNTIME DESTINATION bin
)

#
# Unit test for TritonRepoAgent ... (TODO specify the other classes)
#
set(
  TRITON_REPO_AGENT_SRCS
  ../core/triton_repo_agent.cc
  ../core/status.cc
  ../core/filesystem.cc
  ../core/model_config_utils.cc
  ../core/model_config.cc
  ../core/autofill.cc
)

set(
  TRITON_REPO_AGENT_HDRS
  ../core/triton_repo_agent.h
  ../core/shared_library.h
  ../core/logging.h
  ../core/status.h
  ../core/filesystem.h
  ../core/model_config_utils.h
  ../core/model_config.h
  ${MODEL_CONFIG_PROTO_HDR}
)

set(
  TRITON_REPO_AGENT_TEST_SRCS
  triton_repo_agent_test.cc
  ${TRITON_REPO_AGENT_SRCS}
)

set(
  TRITON_REPO_AGENT_TEST_HDRS
  ${TRITON_REPO_AGENT_HDRS}
)

find_package(GTest REQUIRED)
add_executable(
  triton_repo_agent_test
  ${TRITON_REPO_AGENT_TEST_SRCS}
  ${TRITON_REPO_AGENT_TEST_HDRS}
  $<TARGET_OBJECTS:proto-library>
)
set_target_properties(
  triton_repo_agent_test
  PROPERTIES
    SKIP_BUILD_RPATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH FALSE
    INSTALL_RPATH ""
)
target_include_directories(
  triton_repo_agent_test
  PRIVATE ${GTEST_INCLUDE_DIR}
)
target_link_libraries(
  triton_repo_agent_test
  PRIVATE triton-core-serverapi     # from repo-core
  PRIVATE triton-core-repoagentapi  # from repo-core
  PRIVATE triton-common-error       # from repo-common
  PRIVATE triton-common-json        # from repo-common
  PRIVATE proto-library             # from repo-common
  PRIVATE ${GTEST_LIBRARY}
  PRIVATE ${GTEST_MAIN_LIBRARY}
  PRIVATE protobuf::libprotobuf
  PRIVATE -lpthread
  PRIVATE -ldl
)

# Remove all TRITON_ENABLE_XXX definitions for this test
target_compile_options(triton_repo_agent_test PRIVATE
-UTRITON_ENABLE_ASAN
-UTRITON_ENABLE_NVTX -UTRITON_ENABLE_TRACING
-UTRITON_ENABLE_LOGGING
-UTRITON_ENABLE_STATS
-UTRITON_ENABLE_GPU
-UTRITON_ENABLE_METRICS
-UTRITON_ENABLE_METRICS_GPU
-UTRITON_ENABLE_TENSORFLOW
-UTRITON_ENABLE_PYTHON
-UTRITON_ENABLE_TENSORRT
-UTRITON_ENABLE_ONNXRUNTIME
-UTRITON_ENABLE_ONNXRUNTIME_TENSORRT
-UTRITON_ENABLE_ONNXRUNTIME_OPENVINO
-UTRITON_ENABLE_PYTORCH
-UTRITON_ENABLE_ENSEMBLE
-UTRITON_ENABLE_CUDA_GRAPH
-UTRITON_ENABLE_GCS
-UTRITON_ENABLE_AZURE_STORAGE
-UTRITON_ENABLE_S3)

install(
  TARGETS triton_repo_agent_test
  RUNTIME DESTINATION bin
)

##
## RateLimiter
##
#
#set(
#  RATE_LIMITER_SRCS
#  ../core/rate_limiter.cc
#  ../core/status.cc
#)
#
#set(
#  RATE_LIMITER_HDRS
#  ../core/rate_limiter.h
#  ${MODEL_CONFIG_PROTO_HDR}
#)
#
#set(
#  RATE_LIMITER_TEST_SRCS
#  rate_limiter_test.cc
#  ${RATE_LIMITER_SRCS}
#)
#
#set(
#  RATE_LIMITER_TEST_HDRS
#  ${RATE_LIMITER_HDRS}
#)
#
#add_executable(
#  rate_limiter_test
#  ${RATE_LIMITER_TEST_SRCS}
#  ${RATE_LIMITER_TEST_HDRS}
#  $<TARGET_OBJECTS:proto-library>
#)
#set_target_properties(
#  rate_limiter_test
#  PROPERTIES
#    SKIP_BUILD_RPATH TRUE
#    BUILD_WITH_INSTALL_RPATH TRUE
#    INSTALL_RPATH_USE_LINK_PATH FALSE
#    INSTALL_RPATH ""
#)
#target_include_directories(
#  rate_limiter_test
#  PRIVATE ${GTEST_INCLUDE_DIR}
#)
#if(${TRITON_ENABLE_GPU})
#  target_include_directories(rate_limiter_test PRIVATE ${CUDA_INCLUDE_DIRS})
#endif() # TRITON_ENABLE_GPU
#target_link_libraries(
#  rate_limiter_test
#  PRIVATE triton-common-error     # from repo-common
#  PRIVATE triton-core-serverapi   # from repo-core
#  PRIVATE proto-library           # from repo-common
#  PRIVATE ${GTEST_LIBRARY}
#  PRIVATE ${GTEST_MAIN_LIBRARY}
#  PRIVATE protobuf::libprotobuf
#)
#install(
#  TARGETS rate_limiter_test
#  RUNTIME DESTINATION bin
#)

add_subdirectory(sequence sequence)
add_subdirectory(dyna_sequence dyna_sequence)
add_subdirectory(distributed_addsub distributed_addsub)
add_subdirectory(repoagent/relocation_repoagent repoagent/relocation_repoagent)
