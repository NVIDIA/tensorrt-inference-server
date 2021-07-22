#!/usr/bin/env python3
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

import argparse
import logging
import os.path
import multiprocessing
import pathlib
import platform
import shutil
import subprocess
import sys
import traceback
from distutils.dir_util import copy_tree

#
# Build Triton Inference Server.
#

# By default build.py builds the Triton container. The TRITON_VERSION
# file indicates the Triton version and TRITON_VERSION_MAP is used to
# determine the corresponding container version and upstream container
# version (upstream containers are dependencies required by
# Triton). These versions may be overridden. See docs/build.md for
# more information.

# Map from Triton version to corresponding container and component versions.
#
#   triton version ->
#     (triton container version,
#      upstream container version,
#      ORT version,
#      ORT OpenVINO version (use None to disable OpenVINO in ORT),
#      Standalone OpenVINO version,
#      DCGM version
#     )
#
# Currently the OpenVINO versions used in ORT and standalone must
# match because of the way dlopen works with loading the backends. If
# different versions are used then one backend or the other will
# incorrectly load the other version of the openvino libraries.
#
TRITON_VERSION_MAP = {
    '2.13.0dev': (
        '21.08dev',  # triton container
        '21.06',  # upstream container
        '1.8.0',  # ORT
        '2021.2.200',  # ORT OpenVINO
        '2021.2',  # Standalone OpenVINO
        '2.2.8')  # DCGM version
}

EXAMPLE_BACKENDS = ['identity', 'square', 'repeat']
CORE_BACKENDS = ['tensorrt', 'ensemble']
NONCORE_BACKENDS = [
    'tensorflow1', 'tensorflow2', 'onnxruntime', 'python', 'dali', 'pytorch',
    'openvino', 'fil', 'fastertransformer'
]
EXAMPLE_REPOAGENTS = ['checksum']
FLAGS = None


def log(msg, force=False):
    if force or not FLAGS.quiet:
        try:
            print(msg, file=sys.stderr)
        except Exception:
            print('<failed to log>', file=sys.stderr)


def log_verbose(msg):
    if FLAGS.verbose:
        log(msg, force=True)


def fail(msg):
    fail_if(True, msg)


def target_platform():
    if FLAGS.target_platform is not None:
        return FLAGS.target_platform
    return platform.system().lower()


def fail_if(p, msg):
    if p:
        print('error: {}'.format(msg), file=sys.stderr)
        sys.exit(1)


def mkdir(path):
    log_verbose('mkdir: {}'.format(path))
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def rmdir(path):
    log_verbose('rmdir: {}'.format(path))
    shutil.rmtree(path, ignore_errors=True)


def cpdir(src, dest):
    log_verbose('cpdir: {} -> {}'.format(src, dest))
    copy_tree(src, dest, preserve_symlinks=1)


def untar(targetdir, tarfile):
    log_verbose('untar {} into {}'.format(tarfile, targetdir))
    p = subprocess.Popen(['tar', '--strip-components=1', '-xf', tarfile],
                         cwd=targetdir)
    p.wait()
    fail_if(p.returncode != 0,
            'untar {} into {} failed'.format(tarfile, targetdir))


def gitclone(cwd, repo, tag, subdir):
    # If 'tag' starts with "pull/" then it must be of form
    # "pull/<pr>/head". We just clone at "main" and then fetch the
    # reference onto a new branch we name "tritonbuildref".
    if tag.startswith("pull/"):
        log_verbose('git clone of repo "{}" at ref "{}"'.format(repo, tag))
        p = subprocess.Popen([
            'git', 'clone', '--recursive', '--depth=1', '{}/{}.git'.format(
                FLAGS.github_organization, repo), subdir
        ],
                             cwd=cwd)
        p.wait()
        fail_if(p.returncode != 0,
                'git clone of repo "{}" at branch "main" failed'.format(repo))

        log_verbose('git fetch of ref "{}"'.format(tag))
        p = subprocess.Popen(
            ['git', 'fetch', 'origin', '{}:tritonbuildref'.format(tag)],
            cwd=os.path.join(cwd, subdir))
        p.wait()
        fail_if(p.returncode != 0, 'git fetch of ref "{}" failed'.format(tag))

        log_verbose('git checkout of tritonbuildref')
        p = subprocess.Popen(['git', 'checkout', 'tritonbuildref'],
                             cwd=os.path.join(cwd, subdir))
        p.wait()
        fail_if(p.returncode != 0,
                'git checkout of branch "tritonbuildref" failed')

    else:
        log_verbose('git clone of repo "{}" at tag "{}"'.format(repo, tag))
        p = subprocess.Popen([
            'git', 'clone', '--recursive', '--single-branch', '--depth=1', '-b',
            tag, '{}/{}.git'.format(FLAGS.github_organization, repo), subdir
        ],
                             cwd=cwd)
        p.wait()
        fail_if(p.returncode != 0,
                'git clone of repo "{}" at tag "{}" failed'.format(repo, tag))


def prebuild_command():
    p = subprocess.Popen(FLAGS.container_prebuild_command.split())
    p.wait()
    fail_if(p.returncode != 0, 'container prebuild cmd failed')


def cmake(cwd, args):
    log_verbose('cmake {}'.format(args))
    p = subprocess.Popen([
        'cmake',
    ] + args, cwd=cwd)
    p.wait()
    fail_if(p.returncode != 0, 'cmake failed')


def makeinstall(cwd, target='install'):
    log_verbose('make {}'.format(target))

    if target_platform() == 'windows':
        verbose_flag = '-v:detailed' if FLAGS.verbose else '-clp:ErrorsOnly'
        buildtype_flag = '-p:Configuration={}'.format(FLAGS.build_type)
        p = subprocess.Popen([
            'msbuild.exe', '-m:{}'.format(str(FLAGS.build_parallel)),
            verbose_flag, buildtype_flag, '{}.vcxproj'.format(target)
        ],
                             cwd=cwd)
    else:
        verbose_flag = 'VERBOSE=1' if FLAGS.verbose else 'VERBOSE=0'
        p = subprocess.Popen(
            ['make', '-j',
             str(FLAGS.build_parallel), verbose_flag, target],
            cwd=cwd)

    p.wait()
    fail_if(p.returncode != 0, 'make {} failed'.format(target))


def cmake_enable(flag):
    return 'ON' if flag else 'OFF'


def core_cmake_args(components, backends, install_dir):
    cargs = [
        '-DCMAKE_BUILD_TYPE={}'.format(FLAGS.build_type),
        '-DCMAKE_INSTALL_PREFIX:PATH={}'.format(install_dir),
        '-DTRITON_COMMON_REPO_TAG:STRING={}'.format(components['common']),
        '-DTRITON_CORE_REPO_TAG:STRING={}'.format(components['core']),
        '-DTRITON_BACKEND_REPO_TAG:STRING={}'.format(components['backend']),
        '-DTRITON_THIRD_PARTY_REPO_TAG:STRING={}'.format(
            components['thirdparty'])
    ]

    cargs.append('-DTRITON_ENABLE_LOGGING:BOOL={}'.format(
        cmake_enable(FLAGS.enable_logging)))
    cargs.append('-DTRITON_ENABLE_STATS:BOOL={}'.format(
        cmake_enable(FLAGS.enable_stats)))
    cargs.append('-DTRITON_ENABLE_METRICS:BOOL={}'.format(
        cmake_enable(FLAGS.enable_metrics)))
    cargs.append('-DTRITON_ENABLE_METRICS_GPU:BOOL={}'.format(
        cmake_enable(FLAGS.enable_gpu_metrics)))
    cargs.append('-DTRITON_ENABLE_TRACING:BOOL={}'.format(
        cmake_enable(FLAGS.enable_tracing)))
    cargs.append('-DTRITON_ENABLE_NVTX:BOOL={}'.format(
        cmake_enable(FLAGS.enable_nvtx)))

    cargs.append('-DTRITON_ENABLE_GPU:BOOL={}'.format(
        cmake_enable(FLAGS.enable_gpu)))
    cargs.append('-DTRITON_MIN_COMPUTE_CAPABILITY={}'.format(
        FLAGS.min_compute_capability))

    cargs.append('-DTRITON_ENABLE_GRPC:BOOL={}'.format(
        cmake_enable('grpc' in FLAGS.endpoint)))
    cargs.append('-DTRITON_ENABLE_HTTP:BOOL={}'.format(
        cmake_enable('http' in FLAGS.endpoint)))
    cargs.append('-DTRITON_ENABLE_SAGEMAKER:BOOL={}'.format(
        cmake_enable('sagemaker' in FLAGS.endpoint)))

    cargs.append('-DTRITON_ENABLE_GCS:BOOL={}'.format(
        cmake_enable('gcs' in FLAGS.filesystem)))
    cargs.append('-DTRITON_ENABLE_S3:BOOL={}'.format(
        cmake_enable('s3' in FLAGS.filesystem)))
    cargs.append('-DTRITON_ENABLE_AZURE_STORAGE:BOOL={}'.format(
        cmake_enable('azure_storage' in FLAGS.filesystem)))

    cargs.append('-DTRITON_ENABLE_TENSORFLOW={}'.format(
        cmake_enable(('tensorflow1' in backends) or
                     ('tensorflow2' in backends))))

    for be in (CORE_BACKENDS + NONCORE_BACKENDS):
        if not be.startswith('tensorflow'):
            cargs.append('-DTRITON_ENABLE_{}={}'.format(
                be.upper(), cmake_enable(be in backends)))
        if (be in CORE_BACKENDS) and (be in backends):
            if be == 'tensorrt':
                cargs += tensorrt_cmake_args()
            elif be == 'ensemble':
                pass
            else:
                fail('unknown core backend {}'.format(be))

    # If TRITONBUILD_* is defined in the env then we use it to set
    # corresponding cmake value.
    for evar, eval in os.environ.items():
        if evar.startswith('TRITONBUILD_'):
            cargs.append('-D{}={}'.format(evar[len('TRITONBUILD_'):], eval))

    cargs.append(FLAGS.cmake_dir)
    return cargs


def repoagent_repo(ra):
    return '{}_repository_agent'.format(ra)


def repoagent_cmake_args(images, components, ra, install_dir):
    if ra in EXAMPLE_REPOAGENTS:
        args = []
    else:
        fail('unknown agent {}'.format(ra))

    cargs = args + [
        '-DCMAKE_BUILD_TYPE={}'.format(FLAGS.build_type),
        '-DCMAKE_INSTALL_PREFIX:PATH={}'.format(install_dir),
        '-DTRITON_COMMON_REPO_TAG:STRING={}'.format(components['common']),
        '-DTRITON_CORE_REPO_TAG:STRING={}'.format(components['core'])
    ]

    cargs.append('-DTRITON_ENABLE_GPU:BOOL={}'.format(
        cmake_enable(FLAGS.enable_gpu)))

    # If TRITONBUILD_* is defined in the env then we use it to set
    # corresponding cmake value.
    for evar, eval in os.environ.items():
        if evar.startswith('TRITONBUILD_'):
            cargs.append('-D{}={}'.format(evar[len('TRITONBUILD_'):], eval))

    cargs.append('..')
    return cargs


def backend_repo(be):
    if (be == 'tensorflow1') or (be == 'tensorflow2'):
        return 'tensorflow_backend'
    return '{}_backend'.format(be)


def backend_cmake_args(images, components, be, install_dir, library_paths):
    if be == 'onnxruntime':
        args = onnxruntime_cmake_args(images, library_paths)
    elif be == 'openvino':
        args = openvino_cmake_args()
    elif be == 'tensorflow1':
        args = tensorflow_cmake_args(1, images, library_paths)
    elif be == 'tensorflow2':
        args = tensorflow_cmake_args(2, images, library_paths)
    elif be == 'python':
        args = []
    elif be == 'dali':
        args = dali_cmake_args()
    elif be == 'pytorch':
        args = pytorch_cmake_args(images)
    elif be == 'fil':
        args = fil_cmake_args(images)
    elif be == 'fastertransformer':
        args = []
    elif be in EXAMPLE_BACKENDS:
        args = []
    else:
        fail('unknown backend {}'.format(be))

    cargs = args + [
        '-DCMAKE_BUILD_TYPE={}'.format(FLAGS.build_type),
        '-DCMAKE_INSTALL_PREFIX:PATH={}'.format(install_dir),
        '-DTRITON_COMMON_REPO_TAG:STRING={}'.format(components['common']),
        '-DTRITON_CORE_REPO_TAG:STRING={}'.format(components['core']),
        '-DTRITON_BACKEND_REPO_TAG:STRING={}'.format(components['backend'])
    ]

    cargs.append('-DTRITON_ENABLE_GPU:BOOL={}'.format(
        cmake_enable(FLAGS.enable_gpu)))

    # If TRITONBUILD_* is defined in the env then we use it to set
    # corresponding cmake value.
    for evar, eval in os.environ.items():
        if evar.startswith('TRITONBUILD_'):
            cargs.append('-D{}={}'.format(evar[len('TRITONBUILD_'):], eval))

    cargs.append('..')
    return cargs


def pytorch_cmake_args(images):
    if "pytorch" in images:
        image = images["pytorch"]
    else:
        image = 'nvcr.io/nvidia/pytorch:{}-py3'.format(
            FLAGS.upstream_container_version)
    return [
        '-DTRITON_PYTORCH_DOCKER_IMAGE={}'.format(image),
    ]


def onnxruntime_cmake_args(images, library_paths):
    cargs = [
        '-DTRITON_ENABLE_ONNXRUNTIME_TENSORRT=ON',
        '-DTRITON_BUILD_ONNXRUNTIME_VERSION={}'.format(
            TRITON_VERSION_MAP[FLAGS.version][2])
    ]

    # If platform is jetpack do not use docker based build
    if target_platform() == 'jetpack':
        ort_lib_path = library_paths['onnxruntime'] + "/lib"
        ort_include_path = library_paths['onnxruntime'] + "/include"
        cargs += [
            '-DTRITON_ONNXRUNTIME_INCLUDE_PATHS={}'.format(ort_include_path),
            '-DTRITON_ONNXRUNTIME_LIB_PATHS={}'.format(ort_lib_path),
            '-DTRITON_ENABLE_ONNXRUNTIME_OPENVINO=OFF'
        ]
    else:
        if target_platform() == 'windows':
            if 'base' in images:
                cargs.append('-DTRITON_BUILD_CONTAINER={}'.format(
                    images['base']))
        else:
            if 'base' in images:
                cargs.append('-DTRITON_BUILD_CONTAINER={}'.format(
                    images['base']))
            else:
                cargs.append('-DTRITON_BUILD_CONTAINER_VERSION={}'.format(
                    TRITON_VERSION_MAP[FLAGS.version][1]))

            if TRITON_VERSION_MAP[FLAGS.version][3] is not None:
                cargs.append('-DTRITON_ENABLE_ONNXRUNTIME_OPENVINO=ON')
                cargs.append(
                    '-DTRITON_BUILD_ONNXRUNTIME_OPENVINO_VERSION={}'.format(
                        TRITON_VERSION_MAP[FLAGS.version][3]))

    return cargs


def openvino_cmake_args():
    cargs = [
        '-DTRITON_BUILD_OPENVINO_VERSION={}'.format(
            TRITON_VERSION_MAP[FLAGS.version][4]),
    ]

    if target_platform() == 'windows':
        if 'base' in images:
            cargs.append('-DTRITON_BUILD_CONTAINER={}'.format(images['base']))
    else:
        if 'base' in images:
            cargs.append('-DTRITON_BUILD_CONTAINER={}'.format(images['base']))
        else:
            cargs.append('-DTRITON_BUILD_CONTAINER_VERSION={}'.format(
                TRITON_VERSION_MAP[FLAGS.version][1]))

    return cargs


def tensorrt_cmake_args():
    if target_platform() == 'windows':
        return [
            '-DTRITON_TENSORRT_INCLUDE_PATHS=c:/TensorRT/include',
        ]

    return []


def tensorflow_cmake_args(ver, images, library_paths):
    backend_name = "tensorflow{}".format(ver)

    # If platform is jetpack do not use docker images
    extra_args = []
    if target_platform() == 'jetpack':
        if backend_name in library_paths:
            extra_args = [
                '-DTRITON_TENSORFLOW_LIB_PATHS={}'.format(
                    library_paths[backend_name])
            ]
    else:
        # If a specific TF image is specified use it, otherwise pull from NGC.
        if backend_name in images:
            image = images[backend_name]
        else:
            image = 'nvcr.io/nvidia/tensorflow:{}-tf{}-py3'.format(
                FLAGS.upstream_container_version, ver)
        extra_args = ['-DTRITON_TENSORFLOW_DOCKER_IMAGE={}'.format(image)]
    return ['-DTRITON_TENSORFLOW_VERSION={}'.format(ver)] + extra_args


def dali_cmake_args():
    return [
        '-DTRITON_DALI_SKIP_DOWNLOAD=OFF',
    ]


def install_dcgm_libraries(dcgm_version):
    if dcgm_version == '':
        fail(
            'unable to determine default repo-tag, DCGM version not known for {}'
            .format(FLAGS.version))
        return ''
    else:

        return '''
ENV DCGM_VERSION {}
# Install DCGM. Steps from https://developer.nvidia.com/dcgm#Downloads
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin \
&& mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600 \
&& apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub \
&& add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
RUN apt-get update \
&& apt-get install -y datacenter-gpu-manager=1:{}
'''.format(dcgm_version, dcgm_version)


def fil_cmake_args(images):
    cargs = ['-DTRITON_FIL_DOCKER_BUILD=ON']
    if 'base' in images:
        cargs.append('-DTRITON_BUILD_CONTAINER={}'.format(images['base']))
    else:
        cargs.append('-DTRITON_BUILD_CONTAINER_VERSION={}'.format(
            TRITON_VERSION_MAP[FLAGS.version][1]))

    return cargs


def get_container_versions(version, container_version,
                           upstream_container_version):
    if container_version is None:
        if version not in TRITON_VERSION_MAP:
            fail('container version not known for {}'.format(version))
        container_version = TRITON_VERSION_MAP[version][0]
    if upstream_container_version is None:
        if version not in TRITON_VERSION_MAP:
            fail('upstream container version not known for {}'.format(version))
        upstream_container_version = TRITON_VERSION_MAP[version][1]
    return container_version, upstream_container_version


def create_dockerfile_buildbase(ddir, dockerfile_name, argmap):
    df = '''
ARG TRITON_VERSION={}
ARG TRITON_CONTAINER_VERSION={}
ARG BASE_IMAGE={}
'''.format(argmap['TRITON_VERSION'], argmap['TRITON_CONTAINER_VERSION'],
           argmap['BASE_IMAGE'])

    df += '''
FROM ${BASE_IMAGE}

ARG TRITON_VERSION
ARG TRITON_CONTAINER_VERSION
'''
    # Install the windows- or linux-specific buildbase dependencies
    if target_platform() == 'windows':
        df += '''
SHELL ["cmd", "/S", "/C"]
'''
    else:
        df += '''
# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive

# libcurl4-openSSL-dev is needed for GCS
# python3-dev is needed by Torchvision
# python3-pip and libarchive-dev is needed by python backend
# uuid-dev and pkg-config is needed for Azure Storage
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            autoconf \
            automake \
            build-essential \
            docker.io \
            git \
            libre2-dev \
            libssl-dev \
            libtool \
            libboost-dev \
            libcurl4-openssl-dev \
            libb64-dev \
            patchelf \
            python3-dev \
            python3-pip \
            python3-setuptools \
            rapidjson-dev \
            software-properties-common \
            unzip \
            wget \
            zlib1g-dev \
            libarchive-dev \
            pkg-config \
            uuid-dev && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip && \
    pip3 install --upgrade wheel setuptools docker

# Server build requires recent version of CMake (FetchContent required)
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | \
      gpg --dearmor - |  \
      tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main' && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      cmake-data=3.18.4-0kitware1ubuntu20.04.1 cmake=3.18.4-0kitware1ubuntu20.04.1
'''

    # Copy in the triton source. We remove existing contents first in
    # case the FROM container has something there already.
    if target_platform() == 'windows':
        df += '''
WORKDIR /workspace
RUN rmdir /S/Q * || exit 0
COPY . .
'''
    else:
        df += '''
WORKDIR /workspace
RUN rm -fr *
COPY . .
ENTRYPOINT []
'''
        df += install_dcgm_libraries(argmap['DCGM_VERSION'])

    df += '''
ENV TRITON_SERVER_VERSION ${TRITON_VERSION}
ENV NVIDIA_TRITON_SERVER_VERSION ${TRITON_CONTAINER_VERSION}
'''

    mkdir(ddir)
    with open(os.path.join(ddir, dockerfile_name), "w") as dfile:
        dfile.write(df)


def create_dockerfile_build(ddir, dockerfile_name, backends):
    df = '''
FROM tritonserver_builder_image AS build
FROM tritonserver_buildbase
COPY --from=build /tmp/tritonbuild /tmp/tritonbuild
'''

    if 'onnxruntime' in backends:
        if target_platform() != 'windows':
            df += '''
# Copy ONNX custom op library and model (needed for testing)
RUN if [ -d /tmp/tritonbuild/onnxruntime ]; then \
      cp /tmp/tritonbuild/onnxruntime/install/test/libcustom_op_library.so /workspace/qa/L0_custom_ops/.; \
      cp /tmp/tritonbuild/onnxruntime/install/test/custom_op_test.onnx /workspace/qa/L0_custom_ops/.; \
    fi
'''

    mkdir(ddir)
    with open(os.path.join(ddir, dockerfile_name), "w") as dfile:
        dfile.write(df)


def create_dockerfile_linux(ddir, dockerfile_name, argmap, backends, repoagents,
                            endpoints):
    df = '''
#
# Multistage build.
#
ARG TRITON_VERSION={}
ARG TRITON_CONTAINER_VERSION={}

ARG BASE_IMAGE={}
ARG BUILD_IMAGE=tritonserver_build

############################################################################
##  Build image
############################################################################
FROM ${{BUILD_IMAGE}} AS tritonserver_build

############################################################################
##  Production stage: Create container with just inference server executable
############################################################################
FROM ${{BASE_IMAGE}}
'''.format(argmap['TRITON_VERSION'], argmap['TRITON_CONTAINER_VERSION'],
           argmap['BASE_IMAGE'])

    df += dockerfile_prepare_container_linux(argmap, backends)

    df += '''
WORKDIR /opt/tritonserver
COPY --chown=1000:1000 LICENSE .
COPY --chown=1000:1000 TRITON_VERSION .
COPY --chown=1000:1000 NVIDIA_Deep_Learning_Container_License.pdf .
COPY --chown=1000:1000 --from=tritonserver_build /tmp/tritonbuild/install/bin/tritonserver bin/
COPY --chown=1000:1000 --from=tritonserver_build /tmp/tritonbuild/install/lib/libtritonserver.so lib/
COPY --chown=1000:1000 --from=tritonserver_build /tmp/tritonbuild/install/include/triton/core include/triton/core

# Top-level include/core not copied so --chown does not set it correctly,
# so explicit set on all of include
RUN chown -R triton-server:triton-server include
'''

    for noncore in NONCORE_BACKENDS:
        if noncore in backends:
            df += '''
COPY --chown=1000:1000 --from=tritonserver_build /tmp/tritonbuild/install/backends backends
'''
            break

    if len(repoagents) > 0:
        df += '''
COPY --chown=1000:1000 --from=tritonserver_build /tmp/tritonbuild/install/repoagents repoagents
'''
    # Add feature labels for SageMaker endpoint
    if 'sagemaker' in endpoints:
        df += '''
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true
COPY --chown=1000:1000 --from=tritonserver_build /workspace/build/sagemaker/serve /usr/bin/.
'''
    mkdir(ddir)
    with open(os.path.join(ddir, dockerfile_name), "w") as dfile:
        dfile.write(df)


def dockerfile_prepare_container_linux(argmap, backends):
    # Common steps to produce docker images shared by build.py and compose.py.
    # Sets enviroment variables, installs dependencies and adds entrypoint
    df = '''
ARG TRITON_VERSION
ARG TRITON_CONTAINER_VERSION

ENV TRITON_SERVER_VERSION ${TRITON_VERSION}
ENV NVIDIA_TRITON_SERVER_VERSION ${TRITON_CONTAINER_VERSION}
LABEL com.nvidia.tritonserver.version="${TRITON_SERVER_VERSION}"

ENV PATH /opt/tritonserver/bin:${PATH}
'''
    df += '''
ENV TF_ADJUST_HUE_FUSED         1
ENV TF_ADJUST_SATURATION_FUSED  1
ENV TF_ENABLE_WINOGRAD_NONFUSED 1
ENV TF_AUTOTUNE_THRESHOLD       2

# Create a user that can be used to run triton as
# non-root. Make sure that this user to given ID 1000. All server
# artifacts copied below are assign to this user.
ENV TRITON_SERVER_USER=triton-server
RUN userdel tensorrt-server > /dev/null 2>&1 || true && \
    if ! id -u $TRITON_SERVER_USER > /dev/null 2>&1 ; then \
        useradd $TRITON_SERVER_USER; \
    fi && \
    [ `id -u $TRITON_SERVER_USER` -eq 1000 ] && \
    [ `id -g $TRITON_SERVER_USER` -eq 1000 ]

# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive

# Common dependencies. FIXME (can any of these be conditional? For
# example libcurl only needed for GCS?)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
         libb64-0d \
         libcurl4-openssl-dev \
         libre2-5 && \
    rm -rf /var/lib/apt/lists/*
'''
    df += install_dcgm_libraries(argmap['DCGM_VERSION'])
    # Add dependencies needed for python backend
    if 'python' in backends:
        df += '''
# python3, python3-pip and some pip installs required for the python backend
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
         python3 libarchive-dev \
         python3-pip && \
    pip3 install --upgrade pip && \
    pip3 install --upgrade wheel setuptools && \
    pip3 install --upgrade numpy && \
    rm -rf /var/lib/apt/lists/*
'''
    df += '''
# Extra defensive wiring for CUDA Compat lib
RUN ln -sf ${_CUDA_COMPAT_PATH}/lib.real ${_CUDA_COMPAT_PATH}/lib \
 && echo ${_CUDA_COMPAT_PATH}/lib > /etc/ld.so.conf.d/00-cuda-compat.conf \
 && ldconfig \
 && rm -f ${_CUDA_COMPAT_PATH}/lib

WORKDIR /opt/tritonserver
RUN rm -fr /opt/tritonserver/*
COPY --chown=1000:1000 nvidia_entrypoint.sh .
ENTRYPOINT ["/opt/tritonserver/nvidia_entrypoint.sh"]
'''
    df += '''
ENV NVIDIA_BUILD_ID {}
LABEL com.nvidia.build.id={}
LABEL com.nvidia.build.ref={}
'''.format(argmap['NVIDIA_BUILD_ID'], argmap['NVIDIA_BUILD_ID'],
           argmap['NVIDIA_BUILD_REF'])

    return df


def create_dockerfile_windows(ddir, dockerfile_name, argmap, backends,
                              repoagents):
    df = '''
#
# Multistage build.
#
ARG TRITON_VERSION={}
ARG TRITON_CONTAINER_VERSION={}

ARG BASE_IMAGE={}
ARG BUILD_IMAGE=tritonserver_build

############################################################################
##  Build image
############################################################################
FROM ${{BUILD_IMAGE}} AS tritonserver_build

############################################################################
##  Production stage: Create container with just inference server executable
############################################################################
FROM ${{BASE_IMAGE}}

ARG TRITON_VERSION
ARG TRITON_CONTAINER_VERSION

ENV TRITON_SERVER_VERSION ${{TRITON_VERSION}}
ENV NVIDIA_TRITON_SERVER_VERSION ${{TRITON_CONTAINER_VERSION}}
LABEL com.nvidia.tritonserver.version="${{TRITON_SERVER_VERSION}}"

RUN setx path "%path%;C:\opt\tritonserver\bin"
'''.format(argmap['TRITON_VERSION'], argmap['TRITON_CONTAINER_VERSION'],
           argmap['BASE_IMAGE'])
    df += '''
WORKDIR /opt/tritonserver
RUN rmdir /S/Q * || exit 0
COPY LICENSE .
COPY TRITON_VERSION .
COPY NVIDIA_Deep_Learning_Container_License.pdf .
COPY --from=tritonserver_build /tmp/tritonbuild/install/bin bin
COPY --from=tritonserver_build /tmp/tritonbuild/install/lib/tritonserver.lib lib/
COPY --from=tritonserver_build /tmp/tritonbuild/install/include/triton/core include/triton/core
'''

    for noncore in NONCORE_BACKENDS:
        if noncore in backends:
            df += '''
COPY --from=tritonserver_build /tmp/tritonbuild/install/backends backends
'''
            break

    df += '''
ENTRYPOINT []
ENV NVIDIA_BUILD_ID {}
LABEL com.nvidia.build.id={}
LABEL com.nvidia.build.ref={}
'''.format(argmap['NVIDIA_BUILD_ID'], argmap['NVIDIA_BUILD_ID'],
           argmap['NVIDIA_BUILD_REF'])

    mkdir(ddir)
    with open(os.path.join(ddir, dockerfile_name), "w") as dfile:
        dfile.write(df)


def container_build(images, backends, repoagents, endpoints):
    # The cmake, build and install directories within the container.
    build_dir = os.path.join(os.sep, 'tmp', 'tritonbuild')
    install_dir = os.path.join(os.sep, 'tmp', 'tritonbuild', 'install')
    if target_platform() == 'windows':
        cmake_dir = os.path.normpath('c:/workspace/build')
    else:
        cmake_dir = '/workspace/build'

    # We can't use docker module for building container because it
    # doesn't stream output and it also seems to handle cache-from
    # incorrectly which leads to excessive rebuilds in the multistage
    # build.
    if 'base' in images:
        base_image = images['base']
    elif target_platform() == 'windows':
        base_image = 'mcr.microsoft.com/dotnet/framework/sdk:4.8'
    else:
        base_image = 'nvcr.io/nvidia/tritonserver:{}-py3-min'.format(
            FLAGS.upstream_container_version)

    dockerfileargmap = {
        'NVIDIA_BUILD_REF':
            '' if FLAGS.build_sha is None else FLAGS.build_sha,
        'NVIDIA_BUILD_ID':
            '<unknown>' if FLAGS.build_id is None else FLAGS.build_id,
        'TRITON_VERSION':
            FLAGS.version,
        'TRITON_CONTAINER_VERSION':
            FLAGS.container_version,
        'BASE_IMAGE':
            base_image,
        'DCGM_VERSION':
            '' if FLAGS.version is None or FLAGS.version
            not in TRITON_VERSION_MAP else TRITON_VERSION_MAP[FLAGS.version][5],
    }

    cachefrommap = [
        'tritonserver_buildbase', 'tritonserver_buildbase_cache0',
        'tritonserver_buildbase_cache1'
    ]

    cachefromargs = ['--cache-from={}'.format(k) for k in cachefrommap]
    commonargs = [
        'docker', 'build', '-f',
        os.path.join(FLAGS.build_dir, 'Dockerfile.buildbase')
    ]
    if not FLAGS.no_container_pull:
        commonargs += [
            '--pull',
        ]

    log_verbose('buildbase container {}'.format(commonargs + cachefromargs))
    create_dockerfile_buildbase(FLAGS.build_dir, 'Dockerfile.buildbase',
                                dockerfileargmap)
    try:
        # Create buildbase image, this is an image with all
        # dependencies needed for the build.
        p = subprocess.Popen(commonargs + cachefromargs +
                             ['-t', 'tritonserver_buildbase', '.'])
        p.wait()
        fail_if(p.returncode != 0, 'docker build tritonserver_buildbase failed')

        # Before attempting to run the new image, make sure any
        # previous 'tritonserver_builder' container is removed.
        client = docker.from_env(timeout=3600)

        try:
            existing = client.containers.get('tritonserver_builder')
            existing.remove(force=True)
        except docker.errors.NotFound:
            pass  # ignore

        # Next run build.py inside the container with the same flags
        # as was used to run this instance, except:
        #
        # --no-container-build is added so that within the buildbase
        # container we just created we do not attempt to do a nested
        # container build
        #
        # Add --version, --container-version and
        # --upstream-container-version flags since they can be set
        # automatically and so may not be in sys.argv
        #
        # --cmake-dir is overridden to 'cmake_dir'
        #
        # --build-dir is added/overridden to 'build_dir'
        #
        # --install-dir is added/overridden to 'install_dir'
        runargs = [
            'python3',
            './build.py',
        ]
        runargs += sys.argv[1:]
        runargs += [
            '--no-container-build',
        ]
        if FLAGS.version is not None:
            runargs += ['--version', FLAGS.version]
        if FLAGS.container_version is not None:
            runargs += ['--container-version', FLAGS.container_version]
        if FLAGS.upstream_container_version is not None:
            runargs += [
                '--upstream-container-version', FLAGS.upstream_container_version
            ]

        runargs += ['--cmake-dir', cmake_dir]
        runargs += ['--build-dir', build_dir]
        runargs += ['--install-dir', install_dir]

        dockerrunargs = [
            'docker', 'run', '--name', 'tritonserver_builder', '-w',
            '/workspace'
        ]
        if target_platform() == 'windows':
            dockerrunargs += [
                '-v', '\\\\.\pipe\docker_engine:\\\\.\pipe\docker_engine'
            ]
        else:
            dockerrunargs += ['-v', '/var/run/docker.sock:/var/run/docker.sock']
        dockerrunargs += [
            'tritonserver_buildbase',
        ]
        dockerrunargs += runargs

        log_verbose(dockerrunargs)
        p = subprocess.Popen(dockerrunargs)
        p.wait()
        fail_if(p.returncode != 0, 'docker run tritonserver_builder failed')

        container = client.containers.get('tritonserver_builder')

        # It is possible to copy the install artifacts from the
        # container at this point (and, for example put them in the
        # specified install directory on the host). But for container
        # build we just want to use the artifacts in the server base
        # container which is created below.
        #mkdir(FLAGS.install_dir)
        #tarfilename = os.path.join(FLAGS.install_dir, 'triton.tar')
        #install_tar, stat_tar = container.get_archive(install_dir)
        #with open(tarfilename, 'wb') as taroutfile:
        #    for d in install_tar:
        #        taroutfile.write(d)
        #untar(FLAGS.install_dir, tarfilename)

        # Build is complete, save the container as the
        # tritonserver_build image. We must to this in two steps:
        #
        #   1. Commit the container as image
        #   "tritonserver_builder_image". This image can't be used
        #   directly because it binds the /var/run/docker.sock mount
        #   and so you would need to always run with that mount
        #   specified... so it can be used this way but very
        #   inconvenient.
        #
        #   2. Perform a docker build to create "tritonserver_build"
        #   from "tritonserver_builder_image" that is essentially
        #   identical but removes the mount.
        try:
            client.images.remove('tritonserver_builder_image', force=True)
        except docker.errors.ImageNotFound:
            pass  # ignore

        container.commit('tritonserver_builder_image', 'latest')
        container.remove(force=True)

        create_dockerfile_build(FLAGS.build_dir, 'Dockerfile.build', backends)
        p = subprocess.Popen([
            'docker', 'build', '-t', 'tritonserver_build', '-f',
            os.path.join(FLAGS.build_dir, 'Dockerfile.build'), '.'
        ])
        p.wait()
        fail_if(p.returncode != 0, 'docker build tritonserver_build failed')

        # Final base image... this is a multi-stage build that uses
        # the install artifacts from the tritonserver_build
        # container.
        if target_platform() == 'windows':
            create_dockerfile_windows(FLAGS.build_dir, 'Dockerfile',
                                      dockerfileargmap, backends, repoagents)
        else:
            create_dockerfile_linux(FLAGS.build_dir, 'Dockerfile',
                                    dockerfileargmap, backends, repoagents,
                                    endpoints)
        p = subprocess.Popen([
            'docker', 'build', '-f',
            os.path.join(FLAGS.build_dir, 'Dockerfile')
        ] + ['-t', 'tritonserver', '.'])
        p.wait()
        fail_if(p.returncode != 0, 'docker build tritonserver failed')

    except Exception as e:
        logging.error(traceback.format_exc())
        fail('container build failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    group_qv = parser.add_mutually_exclusive_group()
    group_qv.add_argument('-q',
                          '--quiet',
                          action="store_true",
                          required=False,
                          help='Disable console output.')
    group_qv.add_argument('-v',
                          '--verbose',
                          action="store_true",
                          required=False,
                          help='Enable verbose output.')

    parser.add_argument('--no-container-build',
                        action="store_true",
                        required=False,
                        help='Do not use Docker container for build.')
    parser.add_argument(
        '--no-container-pull',
        action="store_true",
        required=False,
        help='Do not use Docker --pull argument when building container.')
    parser.add_argument(
        '--target-platform',
        required=False,
        default=None,
        help=
        'Target for build, can be "ubuntu", "windows" or "jetpack". If not specified, build targets the current platform.'
    )

    parser.add_argument('--build-id',
                        type=str,
                        required=False,
                        help='Build ID associated with the build.')
    parser.add_argument('--build-sha',
                        type=str,
                        required=False,
                        help='SHA associated with the build.')
    parser.add_argument(
        '--build-dir',
        type=str,
        required=True,
        help=
        'Build directory. All repo clones and builds will be performed in this directory.'
    )
    parser.add_argument(
        '--install-dir',
        type=str,
        required=False,
        default=None,
        help='Install directory, default is <builddir>/opt/tritonserver.')
    parser.add_argument(
        '--cmake-dir',
        type=str,
        required=False,
        help='Directory containing the CMakeLists.txt file for Triton server.')
    parser.add_argument(
        '--library-paths',
        action='append',
        required=False,
        default=None,
        help=
        'Specify library paths for respective backends in build as <backend-name>[:<library_path>].'
    )
    parser.add_argument(
        '--build-type',
        required=False,
        default='Release',
        help=
        'Build type, one of "Release", "Debug", "RelWithDebInfo" or "MinSizeRel". Default is "Release".'
    )
    parser.add_argument(
        '-j',
        '--build-parallel',
        type=int,
        required=False,
        default=None,
        help='Build parallelism. Defaults to 2 * number-of-cores.')

    parser.add_argument(
        '--github-organization',
        type=str,
        required=False,
        default='https://github.com/triton-inference-server',
        help=
        'The GitHub organization containing the repos used for the build. Defaults to "https://github.com/triton-inference-server".'
    )
    parser.add_argument(
        '--version',
        type=str,
        required=False,
        help=
        'The Triton version. If not specified defaults to the value in the TRITON_VERSION file.'
    )
    parser.add_argument(
        '--container-version',
        type=str,
        required=False,
        help=
        'The Triton container version to build. If not specified the container version will be chosen automatically based on --version value.'
    )
    parser.add_argument(
        '--upstream-container-version',
        type=str,
        required=False,
        help=
        'The upstream container version to use for the build. If not specified the upstream container version will be chosen automatically based on --version value.'
    )
    parser.add_argument(
        '--container-prebuild-command',
        type=str,
        required=False,
        help=
        'When performing a container build, this command will be executed within the container just before the build it performed.'
    )
    parser.add_argument(
        '--image',
        action='append',
        required=False,
        help=
        'Use specified Docker image in build as <image-name>,<full-image-name>. <image-name> can be "base", "tensorflow1", "tensorflow2", or "pytorch".'
    )

    parser.add_argument('--enable-logging',
                        action="store_true",
                        required=False,
                        help='Enable logging.')
    parser.add_argument('--enable-stats',
                        action="store_true",
                        required=False,
                        help='Enable statistics collection.')
    parser.add_argument('--enable-metrics',
                        action="store_true",
                        required=False,
                        help='Enable metrics reporting.')
    parser.add_argument('--enable-gpu-metrics',
                        action="store_true",
                        required=False,
                        help='Include GPU metrics in reported metrics.')
    parser.add_argument('--enable-tracing',
                        action="store_true",
                        required=False,
                        help='Enable tracing.')
    parser.add_argument('--enable-nvtx',
                        action="store_true",
                        required=False,
                        help='Enable NVTX.')
    parser.add_argument('--enable-gpu',
                        action="store_true",
                        required=False,
                        help='Enable GPU support.')
    parser.add_argument(
        '--min-compute-capability',
        type=str,
        required=False,
        default='6.0',
        help='Minimum CUDA compute capability supported by server.')

    parser.add_argument(
        '--endpoint',
        action='append',
        required=False,
        help=
        'Include specified endpoint in build. Allowed values are "grpc", "http" and "sagemaker".'
    )
    parser.add_argument(
        '--filesystem',
        action='append',
        required=False,
        help=
        'Include specified filesystem in build. Allowed values are "gcs", "azure_storage" and "s3".'
    )
    parser.add_argument(
        '--backend',
        action='append',
        required=False,
        help=
        'Include specified backend in build as <backend-name>[:<repo-tag>]. If <repo-tag> starts with "pull/" then it refers to a pull-request reference, otherwise <repo-tag> indicates the git tag/branch to use for the build. If the version is non-development then the default <repo-tag> is the release branch matching the container version (e.g. version 21.06 -> branch r21.06); otherwise the default <repo-tag> is "main" (e.g. version 21.06dev -> branch main).'
    )
    parser.add_argument(
        '--repo-tag',
        action='append',
        required=False,
        help=
        'The version of a component to use in the build as <component-name>:<repo-tag>. <component-name> can be "common", "core", "backend" or "thirdparty". If <repo-tag> starts with "pull/" then it refers to a pull-request reference, otherwise <repo-tag> indicates the git tag/branch. If the version is non-development then the default <repo-tag> is the release branch matching the container version (e.g. version 21.06 -> branch r21.06); otherwise the default <repo-tag> is "main" (e.g. version 21.06dev -> branch main).'
    )
    parser.add_argument(
        '--repoagent',
        action='append',
        required=False,
        help=
        'Include specified repo agent in build as <repoagent-name>[:<repo-tag>]. If <repo-tag> starts with "pull/" then it refers to a pull-request reference, otherwise <repo-tag> indicates the git tag/branch to use for the build. If the version is non-development then the default <repo-tag> is the release branch matching the container version (e.g. version 21.06 -> branch r21.06); otherwise the default <repo-tag> is "main" (e.g. version 21.06dev -> branch main).'
    )

    FLAGS = parser.parse_args()

    if FLAGS.image is None:
        FLAGS.image = []
    if FLAGS.repo_tag is None:
        FLAGS.repo_tag = []
    if FLAGS.backend is None:
        FLAGS.backend = []
    if FLAGS.endpoint is None:
        FLAGS.endpoint = []
    if FLAGS.filesystem is None:
        FLAGS.filesystem = []
    if FLAGS.repoagent is None:
        FLAGS.repoagent = []
    if FLAGS.library_paths is None:
        FLAGS.library_paths = []

    # FLAGS.cmake_dir is required for non-container builds. For
    # container builds it is set above to the value appropriate for
    # building within the buildbase container.
    if FLAGS.no_container_build:
        if FLAGS.cmake_dir is None:
            fail('--cmake-dir required for Triton core build')

    # Determine the versions. Start with Triton version, if --version
    # is not explicitly specified read from TRITON_VERSION file.
    if FLAGS.version is None:
        with open('TRITON_VERSION', "r") as vfile:
            FLAGS.version = vfile.readline().strip()

    log('version {}'.format(FLAGS.version))

    # Determine the default repo-tag that should be used for images,
    # backends and repo-agents if a repo-tag is not given
    # explicitly. For release branches we use the release branch as
    # the default, otherwise we use 'main'.
    default_repo_tag = 'main'
    cver = FLAGS.container_version
    if cver is None:
        if FLAGS.version not in TRITON_VERSION_MAP:
            fail(
                'unable to determine default repo-tag, container version not known for {}'
                .format(FLAGS.version))
        cver = TRITON_VERSION_MAP[FLAGS.version][0]
    if not cver.endswith('dev'):
        default_repo_tag = 'r' + cver
    log('default repo-tag: {}'.format(default_repo_tag))

    # For other versions use the TRITON_VERSION_MAP unless explicitly
    # given.
    if not FLAGS.no_container_build:
        FLAGS.container_version, FLAGS.upstream_container_version = get_container_versions(
            FLAGS.version, FLAGS.container_version,
            FLAGS.upstream_container_version)

        log('container version {}'.format(FLAGS.container_version))
        log('upstream container version {}'.format(
            FLAGS.upstream_container_version))

    # Initialize map of backends to build and repo-tag for each.
    backends = {}
    for be in FLAGS.backend:
        parts = be.split(':')
        if len(parts) == 1:
            parts.append(default_repo_tag)
        log('backend "{}" at tag/branch "{}"'.format(parts[0], parts[1]))
        backends[parts[0]] = parts[1]

    # Initialize map of repo agents to build and repo-tag for each.
    repoagents = {}
    for be in FLAGS.repoagent:
        parts = be.split(':')
        if len(parts) == 1:
            parts.append(default_repo_tag)
        log('repoagent "{}" at tag/branch "{}"'.format(parts[0], parts[1]))
        repoagents[parts[0]] = parts[1]

    # Initialize map of docker images.
    images = {}
    for img in FLAGS.image:
        parts = img.split(',')
        fail_if(
            len(parts) != 2,
            '--image must specific <image-name>,<full-image-registry>')
        fail_if(
            parts[0] not in ['base', 'pytorch', 'tensorflow1', 'tensorflow2'],
            'unsupported value for --image')
        log('image "{}": "{}"'.format(parts[0], parts[1]))
        images[parts[0]] = parts[1]

    # Initialize map of library paths for each backend.
    library_paths = {}
    for lpath in FLAGS.library_paths:
        parts = lpath.split(':')
        if len(parts) == 2:
            log('backend "{}" library path "{}"'.format(parts[0], parts[1]))
            library_paths[parts[0]] = parts[1]

    # If --container-build is specified then we perform the actual
    # build within a build container and then from that create a
    # tritonserver container holding the results of the build.
    if not FLAGS.no_container_build:
        import docker
        container_build(images, backends, repoagents, FLAGS.endpoint)
        sys.exit(0)

    # If there is a container pre-build command assume this invocation
    # is being done within the build container and so run the
    # pre-build command.
    if (FLAGS.container_prebuild_command):
        prebuild_command()

    log('Building Triton Inference Server')

    if FLAGS.install_dir is None:
        FLAGS.install_dir = os.path.join(FLAGS.build_dir, "opt", "tritonserver")
    if FLAGS.build_parallel is None:
        FLAGS.build_parallel = multiprocessing.cpu_count() * 2

    # Initialize map of common components and repo-tag for each.
    components = {
        'common': default_repo_tag,
        'core': default_repo_tag,
        'backend': default_repo_tag,
        'thirdparty': default_repo_tag
    }
    for be in FLAGS.repo_tag:
        parts = be.split(':')
        fail_if(
            len(parts) != 2,
            '--repo-tag must specific <component-name>:<repo-tag>')
        fail_if(
            parts[0] not in components,
            '--repo-tag <component-name> must be "common", "core", "backend", or "thirdparty"'
        )
        components[parts[0]] = parts[1]
    for c in components:
        log('component "{}" at tag/branch "{}"'.format(c, components[c]))

    # Build the core server. For now the core is contained in this
    # repo so we just build in place
    if True:
        repo_build_dir = os.path.join(FLAGS.build_dir, 'tritonserver', 'build')
        repo_install_dir = os.path.join(FLAGS.build_dir, 'tritonserver',
                                        'install')

        mkdir(repo_build_dir)
        cmake(repo_build_dir,
              core_cmake_args(components, backends, repo_install_dir))
        makeinstall(repo_build_dir, target='server')

        core_install_dir = FLAGS.install_dir
        mkdir(core_install_dir)
        cpdir(repo_install_dir, core_install_dir)

    # Build each backend...
    for be in backends:
        # Core backends are not built separately from core so skip...
        if (be in CORE_BACKENDS):
            continue

        repo_build_dir = os.path.join(FLAGS.build_dir, be, 'build')
        repo_install_dir = os.path.join(FLAGS.build_dir, be, 'install')

        mkdir(FLAGS.build_dir)
        gitclone(FLAGS.build_dir, backend_repo(be), backends[be], be)
        mkdir(repo_build_dir)
        cmake(
            repo_build_dir,
            backend_cmake_args(images, components, be, repo_install_dir,
                               library_paths))
        makeinstall(repo_build_dir)

        backend_install_dir = os.path.join(FLAGS.install_dir, 'backends', be)
        rmdir(backend_install_dir)
        mkdir(backend_install_dir)
        cpdir(os.path.join(repo_install_dir, 'backends', be),
              backend_install_dir)

    # Build each repo agent...
    for ra in repoagents:
        repo_build_dir = os.path.join(FLAGS.build_dir, ra, 'build')
        repo_install_dir = os.path.join(FLAGS.build_dir, ra, 'install')

        mkdir(FLAGS.build_dir)
        gitclone(FLAGS.build_dir, repoagent_repo(ra), repoagents[ra], ra)
        mkdir(repo_build_dir)
        cmake(repo_build_dir,
              repoagent_cmake_args(images, components, ra, repo_install_dir))
        makeinstall(repo_build_dir)

        repoagent_install_dir = os.path.join(FLAGS.install_dir, 'repoagents',
                                             ra)
        rmdir(repoagent_install_dir)
        mkdir(repoagent_install_dir)
        cpdir(os.path.join(repo_install_dir, 'repoagents', ra),
              repoagent_install_dir)
