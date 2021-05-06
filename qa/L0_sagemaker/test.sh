#!/bin/bash
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

REPO_VERSION=${NVIDIA_TRITON_SERVER_VERSION}
if [ "$#" -ge 1 ]; then
    REPO_VERSION=$1
fi
if [ -z "$REPO_VERSION" ]; then
    echo -e "Repository version must be specified"
    echo -e "\n***\n*** Test Failed\n***"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=0

RET=0

rm -rf models
rm -f *.log
rm -f *.out

DATADIR=/data/inferenceserver/${REPO_VERSION}
SERVER=/opt/tritonserver/bin/tritonserver
SERVER_LOG="./server.log"
# Server args is similar to when docker is run in SageMaker environment,
# except the model repository is not at the designated path "/opt/ml"
SERVER_ARGS="--model-repository `pwd`/models --allow-grpc false --allow-http false --allow-metrics false"
source ../common/util.sh

mkdir models && \
    cp -r $DATADIR/qa_model_repository/onnx_int32_int32_int32 models/model && \
    rm -r models/model/2 && rm -r models/model/3 && \
    sed -i "s/onnx_int32_int32_int32/model/" models/model/config.pbtxt

# Use SageMaker's ping endpoint to check server status
# Wait until server health endpoint shows ready. Sets WAIT_RET to 0 on
# success, 1 on failure
function sagemaker_wait_for_server_ready() {
    local spid="$1"; shift
    local wait_time_secs="${1:-30}"; shift

    WAIT_RET=0

    local wait_secs=$wait_time_secs
    until test $wait_secs -eq 0 ; do
        if ! kill -0 $spid; then
            echo "=== Server not running."
            WAIT_RET=1
            return
        fi

        sleep 1;

        set +e
        code=`curl -s -w %{http_code} localhost:8080/ping`
        set -e
        if [ "$code" == "200" ]; then
            return
        fi

        ((wait_secs--));
    done

    echo "=== Timeout $wait_time_secs secs. Server not ready."
    WAIT_RET=1
}

run_server_nowait
sagemaker_wait_for_server_ready $SERVER_PID 10
if [ "$WAIT_RET" != "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    kill $SERVER_PID || true
    cat $SERVER_LOG
    exit 1
fi

# Ping
set +e
code=`curl -s -w %{http_code} -o ./ping.out localhost:8080/ping`
set -e
if [ "$code" != "200" ]; then
    cat ./ping.out
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

# Inference in minimal setting
set +e
code=`curl -s -w %{http_code} -o ./invocations.out -d'{"inputs":[{"name":"INPUT0","datatype":"INT32","shape":[1,16],"data":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]},{"name":"INPUT1","datatype":"INT32","shape":[1,16],"data":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]}]}' localhost:8080/invocations`
set -e
if [ "$code" != "200" ]; then
    cat ./invocations.out
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi
if [ `grep -c "\[2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32\]" ./invocations.out` != "1" ]; then
    RET=1
fi
if [ `grep -c "\[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\]" ./invocations.out` != "1" ]; then
    RET=1
fi

kill $SERVER_PID
wait $SERVER_PID

# Run server with invalid model and exit-on-error=false
rm models/model/1/*
SERVER_ARGS="--model-repository `pwd`/models --allow-grpc false --allow-http false --allow-metrics false \
             --exit-on-error=false"
run_server_nowait
sleep 5
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

# Ping and expect error code
set +e
code=`curl -s -w %{http_code} -o ./ping.out localhost:8080/ping`
set -e
if [ "$code" == "200" ]; then
    cat ./ping.out
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

kill $SERVER_PID
wait $SERVER_PID

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
