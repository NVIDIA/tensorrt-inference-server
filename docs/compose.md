<!--
# Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
-->

# Customize Triton Container

Two Docker images are available from [NVIDIA GPU Cloud
(NGC)](https://ngc.nvidia.com) that make it possible to easily
construct customized versions of Triton. By customizing Triton you can
significantly reduce the size of the Triton image by removing
functionality that you don't require.

Currently the customization is limited as described below but future
releases will increase the amount of customization that is available.
It is also possible to [build Triton](build.md#building-triton)
from source to get more exact customization.

## Use the compose.py script

The `compose.py` script can be found in the [server repository](https://github.com/triton-inference-server/server). Simply clone the repository and run `compose.py` to create a custom container. Note created container version will depend on the branch that was cloned. For example branch [r21.08](https://github.com/triton-inference-server/server/tree/r21.06) should be used to create a image based on the NGC 21.08 Triton release. 

`compose.py` provides `--backend`, `--repoagent` options that allow you to specify which backends and repository agents to include in the custom image. The `--enable-gpu` flag indicates that you want to create an image that supports NVIDIA GPUs. For example, the following creates a new docker image that contains only the TensorFlow 1 and TensorFlow 2 backends and the checksum repository agent.

Example:
```
python3 compose.py --backend tensorflow1 --backend tensorflow2 --repoagent checksum --enable-gpu
```
will provide a container `tritonserver` locally. You can access the container with
```
$ docker run -it tritonserver:latest
```

Note: If `compose.py` is run on release versions older than r21.08, the resulting container will have DCGM version 2.2.8 installed. This may result in different GPU statistic reporting behavior.

## Build it yourself

If you would like to do what `compose.py` is doing under the hood yourself, you can run `compose.py` with the `--dry-run` option and then modify the `Dockerfile.compose` file to satisfy your needs. 


### Triton with Unsupported and Custom Backends

You can [create and build your own Triton
backend](https://github.com/triton-inference-server/backend).  The
result of that build should be a directory containing your backend
shared library and any additional files required by the
backend. Assuming your backend is called "mybackend" and that the
directory is "./mkbackend", adding the following to the Dockerfile `compose.py`
created will create a Triton image that contains all the supported Triton backends plus your
custom backend.

```
COPY ./mybackend /opt/tritonserver/backends/mybackend
```

You also need to install any additional dependencies required by your
backend as part of the Dockerfile. Then use Docker to create the
image.

```
$ docker build -t tritonserver_custom -f Dockerfile.compose .
```
