// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#include <stdint.h>
#include <unistd.h>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <mutex>
#include <thread>

#include "src/core/logging.h"
#include "src/core/server.h"

namespace {

// The inference server object. Once this server is successfully
// created it does *not* transition back to a nullptr value and it is
// *not* explicitly destructed. Thus we assume that 'server_' can
// always be dereferenced.
nvidia::inferenceserver::InferenceServer* server_ = nullptr;

// Exit thread, status, mutex and cv used to signal the main thread
// that it should close the server and exit. Exit status is -1 when
// server is not exiting, and 0/1 when server should exit.
std::unique_ptr<std::thread> exit_thread_;
int exit_status_ = -1;
std::mutex exit_mu_;
std::condition_variable exit_cv_;

void
SignalHandler(int signum)
{
  // Don't need a mutex here since signals should be disabled while in
  // the handler.
  LOG_INFO << "Interrupt signal (" << signum << ") received.";

  if (exit_thread_ != nullptr)
    return;

  exit_thread_.reset(new std::thread([] {
    bool stop_status = server_->Stop();

    std::unique_lock<std::mutex> lock(exit_mu_);
    exit_status_ = (stop_status) ? 0 : 1;
    exit_cv_.notify_all();
  }));

  exit_thread_->detach();
}

}  // namespace

int
main(int argc, char** argv)
{
  server_ = new nvidia::inferenceserver::InferenceServer();
  bool init_status = server_->Init(argc, argv);
  if (!init_status) {
    exit(1);
  }

  // Trap SIGINT and SIGTERM to allow server to exit gracefully
  signal(SIGINT, SignalHandler);
  signal(SIGTERM, SignalHandler);

  // Watch for changes in the model repository.
  server_->PollModelRepository();

  // Wait until a signal terminates the server...
  while (exit_status_ < 0) {
    std::unique_lock<std::mutex> lock(exit_mu_);
    exit_cv_.wait_for(lock, std::chrono::seconds(1));
  }

  return exit_status_;
}
