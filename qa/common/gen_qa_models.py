# Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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
from builtins import range
import os
import sys
import numpy as np
import gen_ensemble_model_utils as emu

FLAGS = None
np_dtype_string = np.dtype(object)


def np_to_model_dtype(np_dtype):
    if np_dtype == bool:
        return "TYPE_BOOL"
    elif np_dtype == np.int8:
        return "TYPE_INT8"
    elif np_dtype == np.int16:
        return "TYPE_INT16"
    elif np_dtype == np.int32:
        return "TYPE_INT32"
    elif np_dtype == np.int64:
        return "TYPE_INT64"
    elif np_dtype == np.uint8:
        return "TYPE_UINT8"
    elif np_dtype == np.uint16:
        return "TYPE_UINT16"
    elif np_dtype == np.float16:
        return "TYPE_FP16"
    elif np_dtype == np.float32:
        return "TYPE_FP32"
    elif np_dtype == np.float64:
        return "TYPE_FP64"
    elif np_dtype == np_dtype_string:
        return "TYPE_STRING"
    return None


def np_to_tf_dtype(np_dtype):
    if np_dtype == bool:
        return tf.bool
    elif np_dtype == np.int8:
        return tf.int8
    elif np_dtype == np.int16:
        return tf.int16
    elif np_dtype == np.int32:
        return tf.int32
    elif np_dtype == np.int64:
        return tf.int64
    elif np_dtype == np.uint8:
        return tf.uint8
    elif np_dtype == np.uint16:
        return tf.uint16
    elif np_dtype == np.float16:
        return tf.float16
    elif np_dtype == np.float32:
        return tf.float32
    elif np_dtype == np.float64:
        return tf.float64
    elif np_dtype == np_dtype_string:
        return tf.string
    return None


def np_to_trt_dtype(np_dtype):
    if np_dtype == bool:
        return trt.bool
    elif np_dtype == np.int8:
        return trt.int8
    elif np_dtype == np.int32:
        return trt.int32
    elif np_dtype == np.float16:
        return trt.float16
    elif np_dtype == np.float32:
        return trt.float32
    return None


def np_to_onnx_dtype(np_dtype):
    if np_dtype == bool:
        return onnx.TensorProto.BOOL
    elif np_dtype == np.int8:
        return onnx.TensorProto.INT8
    elif np_dtype == np.int16:
        return onnx.TensorProto.INT16
    elif np_dtype == np.int32:
        return onnx.TensorProto.INT32
    elif np_dtype == np.int64:
        return onnx.TensorProto.INT64
    elif np_dtype == np.uint8:
        return onnx.TensorProto.UINT8
    elif np_dtype == np.uint16:
        return onnx.TensorProto.UINT16
    elif np_dtype == np.float16:
        return onnx.TensorProto.FLOAT16
    elif np_dtype == np.float32:
        return onnx.TensorProto.FLOAT
    elif np_dtype == np.float64:
        return onnx.TensorProto.DOUBLE
    elif np_dtype == np_dtype_string:
        return onnx.TensorProto.STRING
    return None


def np_to_torch_dtype(np_dtype):
    if np_dtype == bool:
        return torch.bool
    elif np_dtype == np.int8:
        return torch.int8
    elif np_dtype == np.int16:
        return torch.int16
    elif np_dtype == np.int32:
        return torch.int
    elif np_dtype == np.int64:
        return torch.long
    elif np_dtype == np.uint8:
        return torch.uint8
    elif np_dtype == np.uint16:
        return None  # Not supported in Torch
    elif np_dtype == np.float16:
        return None
    elif np_dtype == np.float32:
        return torch.float
    elif np_dtype == np.float64:
        return torch.double
    elif np_dtype == np_dtype_string:
        return None  # Not supported in Torch


def create_graphdef_modelfile(models_dir,
                              max_batch,
                              model_version,
                              input_shape,
                              output0_shape,
                              output1_shape,
                              input_dtype,
                              output0_dtype,
                              output1_dtype,
                              swap=False):

    if not tu.validate_for_tf_model(input_dtype, output0_dtype, output1_dtype,
                                    input_shape, output0_shape, output1_shape):
        return

    tf_input_dtype = np_to_tf_dtype(input_dtype)
    tf_output0_dtype = np_to_tf_dtype(output0_dtype)
    tf_output1_dtype = np_to_tf_dtype(output1_dtype)

    # Create the model. If non-batching then don't include the batch
    # dimension.
    tf.reset_default_graph()
    if max_batch == 0:
        in0 = tf.placeholder(tf_input_dtype, tu.shape_to_tf_shape(input_shape),
                             "INPUT0")
        in1 = tf.placeholder(tf_input_dtype, tu.shape_to_tf_shape(input_shape),
                             "INPUT1")
    else:
        in0 = tf.placeholder(tf_input_dtype, [
            None,
        ] + tu.shape_to_tf_shape(input_shape), "INPUT0")
        in1 = tf.placeholder(tf_input_dtype, [
            None,
        ] + tu.shape_to_tf_shape(input_shape), "INPUT1")

    # If the input is a string, then convert each string to the
    # equivalent int32 value.
    if tf_input_dtype == tf.string:
        in0 = tf.strings.to_number(in0, tf.int32)
        in1 = tf.strings.to_number(in1, tf.int32)

    add = tf.add(in0, in1, "ADD")
    sub = tf.subtract(in0, in1, "SUB")

    # Cast or convert result to the output dtype.
    if tf_output0_dtype == tf.string:
        cast0 = tf.dtypes.as_string(add if not swap else sub, name="TOSTR0")
    else:
        cast0 = tf.cast(add if not swap else sub, tf_output0_dtype, "CAST0")

    if tf_output1_dtype == tf.string:
        cast1 = tf.dtypes.as_string(sub if not swap else add, name="TOSTR1")
    else:
        cast1 = tf.cast(sub if not swap else add, tf_output1_dtype, "CAST1")

    out0 = tf.identity(cast0, "OUTPUT0")
    out1 = tf.identity(cast1, "OUTPUT1")

    # Use a different model name for the non-batching variant
    model_name = tu.get_model_name(
        "graphdef_nobatch" if max_batch == 0 else "graphdef", input_dtype,
        output0_dtype, output1_dtype)
    model_version_dir = models_dir + "/" + model_name + "/" + str(model_version)

    try:
        os.makedirs(model_version_dir)
    except OSError as ex:
        pass  # ignore existing dir

    with tf.Session() as sess:
        graph_io.write_graph(sess.graph.as_graph_def(),
                             model_version_dir,
                             "model.graphdef",
                             as_text=False)


def create_graphdef_modelconfig(models_dir, max_batch, model_version,
                                input_shape, output0_shape, output1_shape,
                                input_dtype, output0_dtype, output1_dtype,
                                output0_label_cnt, version_policy):

    if not tu.validate_for_tf_model(input_dtype, output0_dtype, output1_dtype,
                                    input_shape, output0_shape, output1_shape):
        return

    # Unpack version policy
    version_policy_str = "{ latest { num_versions: 1 }}"
    if version_policy is not None:
        type, val = version_policy
        if type == 'latest':
            version_policy_str = "{{ latest {{ num_versions: {} }}}}".format(
                val)
        elif type == 'specific':
            version_policy_str = "{{ specific {{ versions: {} }}}}".format(val)
        else:
            version_policy_str = "{ all { }}"

    # Use a different model name for the non-batching variant
    model_name = tu.get_model_name(
        "graphdef_nobatch" if max_batch == 0 else "graphdef", input_dtype,
        output0_dtype, output1_dtype)
    config_dir = models_dir + "/" + model_name
    config = '''
name: "{}"
platform: "tensorflow_graphdef"
max_batch_size: {}
version_policy: {}
input [
  {{
    name: "INPUT0"
    data_type: {}
    dims: [ {} ]
  }},
  {{
    name: "INPUT1"
    data_type: {}
    dims: [ {} ]
  }}
]
output [
  {{
    name: "OUTPUT0"
    data_type: {}
    dims: [ {} ]
    label_filename: "output0_labels.txt"
  }},
  {{
    name: "OUTPUT1"
    data_type: {}
    dims: [ {} ]
  }}
]
'''.format(model_name, max_batch, version_policy_str,
           np_to_model_dtype(input_dtype), tu.shape_to_dims_str(input_shape),
           np_to_model_dtype(input_dtype), tu.shape_to_dims_str(input_shape),
           np_to_model_dtype(output0_dtype),
           tu.shape_to_dims_str(output0_shape),
           np_to_model_dtype(output1_dtype),
           tu.shape_to_dims_str(output1_shape))

    try:
        os.makedirs(config_dir)
    except OSError as ex:
        pass  # ignore existing dir

    with open(config_dir + "/config.pbtxt", "w") as cfile:
        cfile.write(config)

    with open(config_dir + "/output0_labels.txt", "w") as lfile:
        for l in range(output0_label_cnt):
            lfile.write("label" + str(l) + "\n")


def create_savedmodel_modelfile(models_dir,
                                max_batch,
                                model_version,
                                input_shape,
                                output0_shape,
                                output1_shape,
                                input_dtype,
                                output0_dtype,
                                output1_dtype,
                                swap=False):

    if not tu.validate_for_tf_model(input_dtype, output0_dtype, output1_dtype,
                                    input_shape, output0_shape, output1_shape):
        return

    tf_input_dtype = np_to_tf_dtype(input_dtype)
    tf_output0_dtype = np_to_tf_dtype(output0_dtype)
    tf_output1_dtype = np_to_tf_dtype(output1_dtype)

    # Create the model. If non-batching then don't include the batch
    # dimension.
    tf.reset_default_graph()
    if max_batch == 0:
        in0 = tf.placeholder(tf_input_dtype, tu.shape_to_tf_shape(input_shape),
                             "TENSOR_INPUT0")
        in1 = tf.placeholder(tf_input_dtype, tu.shape_to_tf_shape(input_shape),
                             "TENSOR_INPUT1")
    else:
        in0 = tf.placeholder(tf_input_dtype, [
            None,
        ] + tu.shape_to_tf_shape(input_shape), "TENSOR_INPUT0")
        in1 = tf.placeholder(tf_input_dtype, [
            None,
        ] + tu.shape_to_tf_shape(input_shape), "TENSOR_INPUT1")

    # If the input is a string, then convert each string to the
    # equivalent float value.
    if tf_input_dtype == tf.string:
        in0 = tf.strings.to_number(in0, tf.int32)
        in1 = tf.strings.to_number(in1, tf.int32)

    add = tf.add(in0, in1, "ADD")
    sub = tf.subtract(in0, in1, "SUB")

    # Cast or convert result to the output dtype.
    if tf_output0_dtype == tf.string:
        cast0 = tf.dtypes.as_string(add if not swap else sub, name="TOSTR0")
    else:
        cast0 = tf.cast(add if not swap else sub, tf_output0_dtype, "CAST0")

    if tf_output1_dtype == tf.string:
        cast1 = tf.dtypes.as_string(sub if not swap else add, name="TOSTR1")
    else:
        cast1 = tf.cast(sub if not swap else add, tf_output1_dtype, "CAST1")

    out0 = tf.identity(cast0, "TENSOR_OUTPUT0")
    out1 = tf.identity(cast1, "TENSOR_OUTPUT1")

    # Use a different model name for the non-batching variant
    model_name = tu.get_model_name(
        "savedmodel_nobatch" if max_batch == 0 else "savedmodel", input_dtype,
        output0_dtype, output1_dtype)
    model_version_dir = models_dir + "/" + model_name + "/" + str(model_version)

    try:
        os.makedirs(model_version_dir)
    except OSError as ex:
        pass  # ignore existing dir

    with tf.Session() as sess:
        input0_tensor = tf.get_default_graph().get_tensor_by_name(
            "TENSOR_INPUT0:0")
        input1_tensor = tf.get_default_graph().get_tensor_by_name(
            "TENSOR_INPUT1:0")
        output0_tensor = tf.get_default_graph().get_tensor_by_name(
            "TENSOR_OUTPUT0:0")
        output1_tensor = tf.get_default_graph().get_tensor_by_name(
            "TENSOR_OUTPUT1:0")
        tf.saved_model.simple_save(sess,
                                   model_version_dir + "/model.savedmodel",
                                   inputs={
                                       "INPUT0": input0_tensor,
                                       "INPUT1": input1_tensor
                                   },
                                   outputs={
                                       "OUTPUT0": output0_tensor,
                                       "OUTPUT1": output1_tensor
                                   })


def create_savedmodel_modelconfig(models_dir, max_batch, model_version,
                                  input_shape, output0_shape, output1_shape,
                                  input_dtype, output0_dtype, output1_dtype,
                                  output0_label_cnt, version_policy):

    if not tu.validate_for_tf_model(input_dtype, output0_dtype, output1_dtype,
                                    input_shape, output0_shape, output1_shape):
        return

    # Unpack version policy
    version_policy_str = "{ latest { num_versions: 1 }}"
    if version_policy is not None:
        type, val = version_policy
        if type == 'latest':
            version_policy_str = "{{ latest {{ num_versions: {} }}}}".format(
                val)
        elif type == 'specific':
            version_policy_str = "{{ specific {{ versions: {} }}}}".format(val)
        else:
            version_policy_str = "{ all { }}"

    # Use a different model name for the non-batching variant
    model_name = tu.get_model_name(
        "savedmodel_nobatch" if max_batch == 0 else "savedmodel", input_dtype,
        output0_dtype, output1_dtype)
    config_dir = models_dir + "/" + model_name
    config = '''
name: "{}"
platform: "tensorflow_savedmodel"
max_batch_size: {}
version_policy: {}
input [
  {{
    name: "INPUT0"
    data_type: {}
    dims: [ {} ]
  }},
  {{
    name: "INPUT1"
    data_type: {}
    dims: [ {} ]
  }}
]
output [
  {{
    name: "OUTPUT0"
    data_type: {}
    dims: [ {} ]
    label_filename: "output0_labels.txt"
  }},
  {{
    name: "OUTPUT1"
    data_type: {}
    dims: [ {} ]
  }}
]
'''.format(model_name, max_batch, version_policy_str,
           np_to_model_dtype(input_dtype), tu.shape_to_dims_str(input_shape),
           np_to_model_dtype(input_dtype), tu.shape_to_dims_str(input_shape),
           np_to_model_dtype(output0_dtype),
           tu.shape_to_dims_str(output0_shape),
           np_to_model_dtype(output1_dtype),
           tu.shape_to_dims_str(output1_shape))

    try:
        os.makedirs(config_dir)
    except OSError as ex:
        pass  # ignore existing dir

    with open(config_dir + "/config.pbtxt", "w") as cfile:
        cfile.write(config)

    with open(config_dir + "/output0_labels.txt", "w") as lfile:
        for l in range(output0_label_cnt):
            lfile.write("label" + str(l) + "\n")


def create_plan_dynamic_rf_modelfile(models_dir, max_batch, model_version,
                                     input_shape, output0_shape, output1_shape,
                                     input_dtype, output0_dtype, output1_dtype,
                                     swap, min_dim, max_dim):
    trt_input_dtype = np_to_trt_dtype(input_dtype)
    trt_output0_dtype = np_to_trt_dtype(output0_dtype)
    trt_output1_dtype = np_to_trt_dtype(output1_dtype)
    trt_memory_format = trt.TensorFormat.LINEAR

    # Create the model
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    if max_batch == 0:
        input_with_batchsize = [i for i in input_shape]
    else:
        input_with_batchsize = [-1] + [i for i in input_shape]

    in0 = network.add_input("INPUT0", trt_input_dtype, input_with_batchsize)
    in1 = network.add_input("INPUT1", trt_input_dtype, input_with_batchsize)
    add = network.add_elementwise(in0, in1, trt.ElementWiseOperation.SUM)
    sub = network.add_elementwise(in0, in1, trt.ElementWiseOperation.SUB)

    out0 = add if not swap else sub
    out1 = sub if not swap else add

    out0.get_output(0).name = "OUTPUT0"
    out1.get_output(0).name = "OUTPUT1"
    network.mark_output(out0.get_output(0))
    network.mark_output(out1.get_output(0))

    out0.get_output(0).dtype = trt_output0_dtype
    out1.get_output(0).dtype = trt_output1_dtype

    in0.allowed_formats = 1 << int(trt_memory_format)
    in1.allowed_formats = 1 << int(trt_memory_format)
    out0.get_output(0).allowed_formats = 1 << int(trt_memory_format)
    out1.get_output(0).allowed_formats = 1 << int(trt_memory_format)

    if (trt_input_dtype == trt.int8):
        in0.dynamic_range = (-128.0, 127.0)
        in1.dynamic_range = (-128.0, 127.0)
    if (trt_output0_dtype == trt.int8):
        out0.get_output(0).dynamic_range = (-128.0, 127.0)
    if (trt_output1_dtype == trt.int8):
        out1.get_output(0).dynamic_range = (-128.0, 127.0)

    min_shape = []
    opt_shape = []
    max_shape = []
    if max_batch != 0:
        min_shape = min_shape + [1]
        opt_shape = opt_shape + [max(1, max_batch)]
        max_shape = max_shape + [max(1, max_batch)]
    for i in input_shape:
        if i == -1:
            min_shape = min_shape + [min_dim]
            opt_shape = opt_shape + [int((max_dim + min_dim) / 2)]
            max_shape = max_shape + [max_dim]
        else:
            min_shape = min_shape + [i]
            opt_shape = opt_shape + [i]
            max_shape = max_shape + [i]

    profile = builder.create_optimization_profile()
    profile.set_shape("INPUT0", min_shape, opt_shape, max_shape)
    profile.set_shape("INPUT1", min_shape, opt_shape, max_shape)
    flags = 1 << int(trt.BuilderFlag.STRICT_TYPES)
    datatype_set = set([trt_input_dtype, trt_output0_dtype, trt_output1_dtype])
    for dt in datatype_set:
        if (dt == trt.int8):
            flags |= 1 << int(trt.BuilderFlag.INT8)
        elif (dt == trt.float16):
            flags |= 1 << int(trt.BuilderFlag.FP16)
    config = builder.create_builder_config()
    config.flags = flags
    config.add_optimization_profile(profile)
    config.max_workspace_size = 1 << 20
    engine = builder.build_engine(network, config)

    # Use a different model name for different kinds of models
    model_name = tu.get_model_name("plan_nobatch" if max_batch == 0 else "plan",
                                   input_dtype, output0_dtype, output1_dtype)
    if min_dim != 1 or max_dim != 32:
        model_name = "{}-{}-{}".format(model_name, min_dim, max_dim)

    model_version_dir = models_dir + "/" + model_name + "/" + str(model_version)

    try:
        os.makedirs(model_version_dir)
    except OSError as ex:
        pass  # ignore existing dir

    with open(model_version_dir + "/model.plan", "wb") as f:
        f.write(engine.serialize())

    del engine
    del builder


def create_plan_dynamic_modelfile(models_dir, max_batch, model_version,
                                  input_shape, output0_shape, output1_shape,
                                  input_dtype, output0_dtype, output1_dtype,
                                  swap, min_dim, max_dim):
    trt_input_dtype = np_to_trt_dtype(input_dtype)
    trt_output0_dtype = np_to_trt_dtype(output0_dtype)
    trt_output1_dtype = np_to_trt_dtype(output1_dtype)

    # Create the model
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    if max_batch == 0:
        input_with_batchsize = [i for i in input_shape]
    else:
        input_with_batchsize = [-1] + [i for i in input_shape]

    in0 = network.add_input("INPUT0", trt_input_dtype, input_with_batchsize)
    in1 = network.add_input("INPUT1", trt_input_dtype, input_with_batchsize)
    add = network.add_elementwise(in0, in1, trt.ElementWiseOperation.SUM)
    sub = network.add_elementwise(in0, in1, trt.ElementWiseOperation.SUB)

    out0 = add if not swap else sub
    out1 = sub if not swap else add

    out0.get_output(0).name = "OUTPUT0"
    out1.get_output(0).name = "OUTPUT1"
    network.mark_output(out0.get_output(0))
    network.mark_output(out1.get_output(0))

    min_shape = []
    opt_shape = []
    max_shape = []
    for i in input_shape:
        if i == -1:
            min_shape = min_shape + [min_dim]
            opt_shape = opt_shape + [int((max_dim + min_dim) / 2)]
            max_shape = max_shape + [max_dim]
        else:
            min_shape = min_shape + [i]
            opt_shape = opt_shape + [i]
            max_shape = max_shape + [i]

    config = builder.create_builder_config()
    # create multiple profiles with same shape for testing
    # with decreasing batch sizes
    profile = []
    for i in range(4):
        profile.append(builder.create_optimization_profile())
        if max_batch == 0:
            profile[i].set_shape("INPUT0", min_shape, opt_shape, max_shape)
            profile[i].set_shape("INPUT1", min_shape, opt_shape, max_shape)
        else:
            bs = [max_batch - i if max_batch > i else 1]
            opt_bs = [1 + i if 1 + i < max_batch - 1 else max_batch - 1]
            # Hardcoded 'max_shape[0] += 1' in default profile for
            # L0_trt_dynamic_shape, to differentiate whether default profile
            # is used if no profile is specified
            max_shape_override = max_shape
            if i == 0 and (min_dim == 1 and max_dim == 32):
                max_shape_override[0] += 1

            profile[i].set_shape("INPUT0", [1] + min_shape, opt_bs + opt_shape,
                                 bs + max_shape_override)
            profile[i].set_shape("INPUT1", [1] + min_shape, opt_bs + opt_shape,
                                 bs + max_shape_override)
        config.add_optimization_profile(profile[i])
    # some profiles with non-one min shape for first dim to test autofiller
    for i in range(2):
        profile.append(builder.create_optimization_profile())
        if max_batch == 0:
            profile[i + 4].set_shape("INPUT0", min_shape, opt_shape, max_shape)
            profile[i + 4].set_shape("INPUT1", min_shape, opt_shape, max_shape)
        else:
            profile[i + 4].set_shape("INPUT0", [5 + i] + min_shape,
                                     [6] + opt_shape, [max_batch] + max_shape)
            profile[i + 4].set_shape("INPUT1", [5 + i] + min_shape,
                                     [6] + opt_shape, [max_batch] + max_shape)
        config.add_optimization_profile(profile[i + 4])
    # Will repeat another profile with same min and max shapes as the first profile to test non-zero profile
    # for infer_variable test.
    profile.append(builder.create_optimization_profile())
    if max_batch == 0:
        profile[6].set_shape("INPUT0", min_shape, opt_shape, max_shape)
        profile[6].set_shape("INPUT1", min_shape, opt_shape, max_shape)
    else:
        profile[6].set_shape("INPUT0", [1] + min_shape, [1] + opt_shape,
                             [max_batch] + max_shape)
        profile[6].set_shape("INPUT1", [1] + min_shape, [1] + opt_shape,
                             [max_batch] + max_shape)
    config.add_optimization_profile(profile[6])

    # Will add some profiles with static shapes to test the cases where min_shape=opt_shape=max_shape
    for i in range(3):
        profile.append(builder.create_optimization_profile())
        if max_batch == 0:
            static_shape = max_shape
            profile[7 + i].set_shape("INPUT0", static_shape, static_shape,
                                     static_shape)
            profile[7 + i].set_shape("INPUT1", static_shape, static_shape,
                                     static_shape)
        else:
            # Skipping alternate batch sizes for testing unsupported batches in L0_trt_dynamic_shape.
            full_static_shape = [1 + (2 * i)] + max_shape
            profile[7 + i].set_shape("INPUT0", full_static_shape,
                                     full_static_shape, full_static_shape)
            profile[7 + i].set_shape("INPUT1", full_static_shape,
                                     full_static_shape, full_static_shape)
        config.add_optimization_profile(profile[7 + i])

    config.max_workspace_size = 1 << 20
    engine = builder.build_engine(network, config)

    # Use a different model name for different kinds of models
    model_name = tu.get_model_name("plan_nobatch" if max_batch == 0 else "plan",
                                   input_dtype, output0_dtype, output1_dtype)
    if min_dim != 1 or max_dim != 32:
        model_name = "{}-{}-{}".format(model_name, min_dim, max_dim)

    model_version_dir = models_dir + "/" + model_name + "/" + str(model_version)

    try:
        os.makedirs(model_version_dir)
    except OSError as ex:
        pass  # ignore existing dir

    with open(model_version_dir + "/model.plan", "wb") as f:
        f.write(engine.serialize())

    del engine
    del builder


def create_plan_fixed_rf_modelfile(models_dir, max_batch, model_version,
                                   input_shape, output0_shape, output1_shape,
                                   input_dtype, output0_dtype, output1_dtype,
                                   swap):
    trt_input_dtype = np_to_trt_dtype(input_dtype)
    trt_output0_dtype = np_to_trt_dtype(output0_dtype)
    trt_output1_dtype = np_to_trt_dtype(output1_dtype)
    trt_memory_format = trt.TensorFormat.LINEAR

    # Create the model
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()
    in0 = network.add_input("INPUT0", trt_input_dtype, input_shape)
    in1 = network.add_input("INPUT1", trt_input_dtype, input_shape)
    add = network.add_elementwise(in0, in1, trt.ElementWiseOperation.SUM)
    sub = network.add_elementwise(in0, in1, trt.ElementWiseOperation.SUB)

    out0 = add if not swap else sub
    out1 = sub if not swap else add

    out0.get_output(0).name = "OUTPUT0"
    out1.get_output(0).name = "OUTPUT1"
    network.mark_output(out0.get_output(0))
    network.mark_output(out1.get_output(0))

    out0.get_output(0).dtype = trt_output0_dtype
    out1.get_output(0).dtype = trt_output1_dtype

    in0.allowed_formats = 1 << int(trt_memory_format)
    in1.allowed_formats = 1 << int(trt_memory_format)
    out0.get_output(0).allowed_formats = 1 << int(trt_memory_format)
    out1.get_output(0).allowed_formats = 1 << int(trt_memory_format)

    if (trt_input_dtype == trt.int8):
        in0.dynamic_range = (-128.0, 127.0)
        in1.dynamic_range = (-128.0, 127.0)
    if (trt_output0_dtype == trt.int8):
        out0.get_output(0).dynamic_range = (-128.0, 127.0)
    if (trt_output1_dtype == trt.int8):
        out1.get_output(0).dynamic_range = (-128.0, 127.0)

    flags = 1 << int(trt.BuilderFlag.STRICT_TYPES)
    datatype_set = set([trt_input_dtype, trt_output0_dtype, trt_output1_dtype])
    for dt in datatype_set:
        if (dt == trt.int8):
            flags |= 1 << int(trt.BuilderFlag.INT8)
        elif (dt == trt.float16):
            flags |= 1 << int(trt.BuilderFlag.FP16)
    config = builder.create_builder_config()
    config.flags = flags
    config.max_workspace_size = 1 << 20
    builder.max_batch_size = max(1, max_batch)
    engine = builder.build_engine(network, config)

    model_name = tu.get_model_name("plan_nobatch" if max_batch == 0 else "plan",
                                   input_dtype, output0_dtype, output1_dtype)
    model_version_dir = models_dir + "/" + model_name + "/" + str(model_version)

    try:
        os.makedirs(model_version_dir)
    except OSError as ex:
        pass  # ignore existing dir

    with open(model_version_dir + "/model.plan", "wb") as f:
        f.write(engine.serialize())

    del engine
    del builder


def create_plan_fixed_modelfile(models_dir, max_batch, model_version,
                                input_shape, output0_shape, output1_shape,
                                input_dtype, output0_dtype, output1_dtype,
                                swap):
    trt_input_dtype = np_to_trt_dtype(input_dtype)
    trt_output0_dtype = np_to_trt_dtype(output0_dtype)
    trt_output1_dtype = np_to_trt_dtype(output1_dtype)

    # Create the model
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()
    in0 = network.add_input("INPUT0", trt_input_dtype, input_shape)
    in1 = network.add_input("INPUT1", trt_input_dtype, input_shape)
    add = network.add_elementwise(in0, in1, trt.ElementWiseOperation.SUM)
    sub = network.add_elementwise(in0, in1, trt.ElementWiseOperation.SUB)

    out0 = add if not swap else sub
    out1 = sub if not swap else add

    out0.get_output(0).name = "OUTPUT0"
    out1.get_output(0).name = "OUTPUT1"
    network.mark_output(out0.get_output(0))
    network.mark_output(out1.get_output(0))

    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 20
    builder.max_batch_size = max(1, max_batch)
    engine = builder.build_engine(network, config)
    del network

    model_name = tu.get_model_name("plan_nobatch" if max_batch == 0 else "plan",
                                   input_dtype, output0_dtype, output1_dtype)
    model_version_dir = models_dir + "/" + model_name + "/" + str(model_version)

    try:
        os.makedirs(model_version_dir)
    except OSError as ex:
        pass  # ignore existing dir

    with open(model_version_dir + "/model.plan", "wb") as f:
        f.write(engine.serialize())

    del engine
    del builder


def create_plan_modelfile(models_dir,
                          max_batch,
                          model_version,
                          input_shape,
                          output0_shape,
                          output1_shape,
                          input_dtype,
                          output0_dtype,
                          output1_dtype,
                          swap=False,
                          min_dim=1,
                          max_dim=32):

    if not tu.validate_for_trt_model(input_dtype, output0_dtype, output1_dtype,
                                     input_shape, output0_shape, output1_shape):
        return

    if input_dtype != np.float32 or output0_dtype != np.float32 or output1_dtype != np.float32:
        if (not tu.shape_is_fixed(input_shape) or
                not tu.shape_is_fixed(output0_shape) or
                not tu.shape_is_fixed(output1_shape)):
            create_plan_dynamic_rf_modelfile(models_dir, max_batch,
                                             model_version, input_shape,
                                             output0_shape, output1_shape,
                                             input_dtype, output0_dtype,
                                             output1_dtype, swap, min_dim,
                                             max_dim)
        else:
            create_plan_fixed_rf_modelfile(models_dir, max_batch, model_version,
                                           input_shape, output0_shape,
                                           output1_shape, input_dtype,
                                           output0_dtype, output1_dtype, swap)

    else:
        if (not tu.shape_is_fixed(input_shape) or
                not tu.shape_is_fixed(output0_shape) or
                not tu.shape_is_fixed(output1_shape)):
            create_plan_dynamic_modelfile(models_dir, max_batch, model_version,
                                          input_shape, output0_shape,
                                          output1_shape, input_dtype,
                                          output0_dtype, output1_dtype, swap,
                                          min_dim, max_dim)
        else:
            create_plan_fixed_modelfile(models_dir, max_batch, model_version,
                                        input_shape, output0_shape,
                                        output1_shape, input_dtype,
                                        output0_dtype, output1_dtype, swap)


def create_plan_modelconfig(models_dir,
                            max_batch,
                            model_version,
                            input_shape,
                            output0_shape,
                            output1_shape,
                            input_dtype,
                            output0_dtype,
                            output1_dtype,
                            output0_label_cnt,
                            version_policy,
                            min_dim=1,
                            max_dim=32):

    if not tu.validate_for_trt_model(input_dtype, output0_dtype, output1_dtype,
                                     input_shape, output0_shape, output1_shape):
        return

    # Unpack version policy
    version_policy_str = "{ latest { num_versions: 1 }}"
    if version_policy is not None:
        type, val = version_policy
        if type == 'latest':
            version_policy_str = "{{ latest {{ num_versions: {} }}}}".format(
                val)
        elif type == 'specific':
            version_policy_str = "{{ specific {{ versions: {} }}}}".format(val)
        else:
            version_policy_str = "{ all { }}"

    # Use a different model name for different kinds of models
    model_name = tu.get_model_name("plan_nobatch" if max_batch == 0 else "plan",
                                   input_dtype, output0_dtype, output1_dtype)
    if min_dim != 1 or max_dim != 32:
        model_name = "{}-{}-{}".format(model_name, min_dim, max_dim)

    config_dir = models_dir + "/" + model_name
    if -1 in input_shape:
        # Selects the sixth profile for FP32 datatype
        # Note the min and max shapes of first and sixth
        # profile are identical.
        profile_index = 6 if input_dtype == np.float32 else 0
        config = '''
name: "{}"
platform: "tensorrt_plan"
max_batch_size: {}
version_policy: {}
input [
  {{
    name: "INPUT0"
    data_type: {}
    dims: [ {} ]
  }},
  {{
    name: "INPUT1"
    data_type: {}
    dims: [ {} ]
  }}
]
output [
  {{
    name: "OUTPUT0"
    data_type: {}
    dims: [ {} ]
    label_filename: "output0_labels.txt"
   }},
  {{
    name: "OUTPUT1"
    data_type: {}
    dims: [ {} ]
  }}
]
instance_group [
  {{
      profile:"{}"
  }}
]
'''.format(model_name, max_batch, version_policy_str,
           np_to_model_dtype(input_dtype), tu.shape_to_dims_str(input_shape),
           np_to_model_dtype(input_dtype), tu.shape_to_dims_str(input_shape),
           np_to_model_dtype(output0_dtype),
           tu.shape_to_dims_str(output0_shape),
           np_to_model_dtype(output1_dtype),
           tu.shape_to_dims_str(output1_shape), profile_index)
    else:
        config = '''
name: "{}"
platform: "tensorrt_plan"
max_batch_size: {}
version_policy: {}
input [
  {{
    name: "INPUT0"
    data_type: {}
    dims: [ {} ]
  }},
  {{
    name: "INPUT1"
    data_type: {}
    dims: [ {} ]
  }}
]
output [
  {{
    name: "OUTPUT0"
    data_type: {}
    dims: [ {} ]
    label_filename: "output0_labels.txt"
   }},
  {{
    name: "OUTPUT1"
    data_type: {}
    dims: [ {} ]
  }}
]
'''.format(model_name, max_batch, version_policy_str,
           np_to_model_dtype(input_dtype), tu.shape_to_dims_str(input_shape),
           np_to_model_dtype(input_dtype), tu.shape_to_dims_str(input_shape),
           np_to_model_dtype(output0_dtype),
           tu.shape_to_dims_str(output0_shape),
           np_to_model_dtype(output1_dtype),
           tu.shape_to_dims_str(output1_shape))

    try:
        os.makedirs(config_dir)
    except OSError as ex:
        pass  # ignore existing dir

    with open(config_dir + "/config.pbtxt", "w") as cfile:
        cfile.write(config)

    with open(config_dir + "/output0_labels.txt", "w") as lfile:
        for l in range(output0_label_cnt):
            lfile.write("label" + str(l) + "\n")


def create_onnx_modelfile(models_dir,
                          max_batch,
                          model_version,
                          input_shape,
                          output0_shape,
                          output1_shape,
                          input_dtype,
                          output0_dtype,
                          output1_dtype,
                          swap=False):

    if not tu.validate_for_onnx_model(input_dtype, output0_dtype, output1_dtype,
                                      input_shape, output0_shape,
                                      output1_shape):
        return

    onnx_input_dtype = np_to_onnx_dtype(input_dtype)
    onnx_output0_dtype = np_to_onnx_dtype(output0_dtype)
    onnx_output1_dtype = np_to_onnx_dtype(output1_dtype)

    onnx_input_shape, idx = tu.shape_to_onnx_shape(input_shape, 0)
    onnx_output0_shape, idx = tu.shape_to_onnx_shape(input_shape, idx)
    onnx_output1_shape, idx = tu.shape_to_onnx_shape(input_shape, idx)

    # Create the model
    model_name = tu.get_model_name("onnx_nobatch" if max_batch == 0 else "onnx",
                                   input_dtype, output0_dtype, output1_dtype)
    model_version_dir = models_dir + "/" + model_name + "/" + str(model_version)

    batch_dim = [] if max_batch == 0 else [None]

    in0 = onnx.helper.make_tensor_value_info("INPUT0", onnx_input_dtype,
                                             batch_dim + onnx_input_shape)
    in1 = onnx.helper.make_tensor_value_info("INPUT1", onnx_input_dtype,
                                             batch_dim + onnx_input_shape)

    out0 = onnx.helper.make_tensor_value_info("OUTPUT0", onnx_output0_dtype,
                                              batch_dim + onnx_output0_shape)
    out1 = onnx.helper.make_tensor_value_info("OUTPUT1", onnx_output1_dtype,
                                              batch_dim + onnx_output1_shape)

    internal_in0 = onnx.helper.make_node("Identity", ["INPUT0"], ["_INPUT0"])
    internal_in1 = onnx.helper.make_node("Identity", ["INPUT1"], ["_INPUT1"])

    # cast int8, int16 input to higer precision int as Onnx Add/Sub operator doesn't support those type
    # Also casting String data type to int32
    if ((onnx_input_dtype == onnx.TensorProto.INT8) or
        (onnx_input_dtype == onnx.TensorProto.INT16) or
        (onnx_input_dtype == onnx.TensorProto.STRING)):
        internal_in0 = onnx.helper.make_node("Cast", ["INPUT0"], ["_INPUT0"],
                                             to=onnx.TensorProto.INT32)
        internal_in1 = onnx.helper.make_node("Cast", ["INPUT1"], ["_INPUT1"],
                                             to=onnx.TensorProto.INT32)

    add = onnx.helper.make_node("Add", ["_INPUT0", "_INPUT1"],
                                ["CAST0" if not swap else "CAST1"])
    sub = onnx.helper.make_node("Sub", ["_INPUT0", "_INPUT1"],
                                ["CAST1" if not swap else "CAST0"])
    cast0 = onnx.helper.make_node("Cast", ["CAST0"], ["OUTPUT0"],
                                  to=onnx_output0_dtype)
    cast1 = onnx.helper.make_node("Cast", ["CAST1"], ["OUTPUT1"],
                                  to=onnx_output1_dtype)

    # Avoid cast from float16 to float16
    # (bug in Onnx Runtime, cast from float16 to float16 will become cast from float16 to float32)
    if onnx_input_dtype == onnx.TensorProto.FLOAT16:
        if onnx_output0_dtype == onnx_input_dtype:
            cast0 = onnx.helper.make_node("Identity", ["CAST0"], ["OUTPUT0"])
        if onnx_output1_dtype == onnx_input_dtype:
            cast1 = onnx.helper.make_node("Identity", ["CAST1"], ["OUTPUT1"])

    onnx_nodes = [internal_in0, internal_in1, add, sub, cast0, cast1]
    onnx_inputs = [in0, in1]
    onnx_outputs = [out0, out1]

    graph_proto = onnx.helper.make_graph(onnx_nodes, model_name, onnx_inputs,
                                         onnx_outputs)
    if FLAGS.onnx_opset > 0:
        model_opset = onnx.helper.make_operatorsetid("", FLAGS.onnx_opset)
        model_def = onnx.helper.make_model(graph_proto,
                                           producer_name="triton",
                                           opset_imports=[model_opset])
    else:
        model_def = onnx.helper.make_model(graph_proto, producer_name="triton")

    try:
        os.makedirs(model_version_dir)
    except OSError as ex:
        pass  # ignore existing dir

    onnx.save(model_def, model_version_dir + "/model.onnx")


def create_onnx_modelconfig(models_dir, max_batch, model_version, input_shape,
                            output0_shape, output1_shape, input_dtype,
                            output0_dtype, output1_dtype, output0_label_cnt,
                            version_policy):

    if not tu.validate_for_onnx_model(input_dtype, output0_dtype, output1_dtype,
                                      input_shape, output0_shape,
                                      output1_shape):
        return

    # Use a different model name for the non-batching variant
    model_name = tu.get_model_name("onnx_nobatch" if max_batch == 0 else "onnx",
                                   input_dtype, output0_dtype, output1_dtype)
    config_dir = models_dir + "/" + model_name

    # [TODO] move create_general_modelconfig() out of emu as it is general
    # enough for all backends to use
    config = emu.create_general_modelconfig(model_name,
                                            "onnxruntime_onnx",
                                            max_batch,
                                            emu.repeat(input_dtype, 2),
                                            emu.repeat(input_shape, 2),
                                            emu.repeat(None, 2),
                                            [output0_dtype, output1_dtype],
                                            [output0_shape, output1_shape],
                                            emu.repeat(None, 2),
                                            ["output0_labels.txt", None],
                                            version_policy=version_policy,
                                            force_tensor_number_suffix=True)

    try:
        os.makedirs(config_dir)
    except OSError as ex:
        pass  # ignore existing dir

    with open(config_dir + "/config.pbtxt", "w") as cfile:
        cfile.write(config)

    with open(config_dir + "/output0_labels.txt", "w") as lfile:
        for l in range(output0_label_cnt):
            lfile.write("label" + str(l) + "\n")


def create_libtorch_modelfile(models_dir,
                              max_batch,
                              model_version,
                              input_shape,
                              output0_shape,
                              output1_shape,
                              input_dtype,
                              output0_dtype,
                              output1_dtype,
                              swap=False):

    if not tu.validate_for_libtorch_model(input_dtype, output0_dtype,
                                          output1_dtype, input_shape,
                                          output0_shape, output1_shape):
        return

    torch_input_dtype = np_to_torch_dtype(input_dtype)
    torch_output0_dtype = np_to_torch_dtype(output0_dtype)
    torch_output1_dtype = np_to_torch_dtype(output1_dtype)

    model_name = tu.get_model_name(
        "libtorch_nobatch" if max_batch == 0 else "libtorch", input_dtype,
        output0_dtype, output1_dtype)
    # handle for -1 (when variable) since can't create tensor with shape of [-1]
    input_shape = [abs(ips) for ips in input_shape]
    # Create the model
    if not swap:

        class AddSubNet(nn.Module):

            def __init__(self, *args):
                self.torch_output0_dtype = args[0][0]
                self.torch_output1_dtype = args[0][1]
                super(AddSubNet, self).__init__()

            def forward(self, input0, input1):
                return (input0 + input1).to(self.torch_output0_dtype), \
                    (input0 - input1).to(self.torch_output1_dtype)

        addSubModel = AddSubNet((torch_output0_dtype, torch_output1_dtype))
        example_input = torch.zeros(input_shape, dtype=torch_input_dtype)
        traced = torch.jit.trace(addSubModel, (example_input, example_input))
    else:

        class SubAddNet(nn.Module):

            def __init__(self, *args):
                self.torch_output0_dtype = args[0][0]
                self.torch_output1_dtype = args[0][1]
                super(SubAddNet, self).__init__()

            def forward(self, input0, input1):
                return (input0 - input1).to(self.torch_output0_dtype), \
                    (input0 + input1).to(self.torch_output1_dtype)

        subAddModel = SubAddNet((torch_output0_dtype, torch_output1_dtype))
        example_input = torch.zeros(input_shape, dtype=torch_input_dtype)
        traced = torch.jit.trace(subAddModel, (example_input, example_input))

    model_version_dir = models_dir + "/" + model_name + "/" + str(model_version)

    try:
        os.makedirs(model_version_dir)
    except OSError as ex:
        pass  # ignore existing dir

    traced.save(model_version_dir + "/model.pt")


def create_libtorch_modelconfig(models_dir, max_batch, model_version,
                                input_shape, output0_shape, output1_shape,
                                input_dtype, output0_dtype, output1_dtype,
                                output0_label_cnt, version_policy):

    if not tu.validate_for_libtorch_model(input_dtype, output0_dtype,
                                          output1_dtype, input_shape,
                                          output0_shape, output1_shape):
        return

    # Unpack version policy
    version_policy_str = "{ latest { num_versions: 1 }}"
    if version_policy is not None:
        type, val = version_policy
        if type == 'latest':
            version_policy_str = "{{ latest {{ num_versions: {} }}}}".format(
                val)
        elif type == 'specific':
            version_policy_str = "{{ specific {{ versions: {} }}}}".format(val)
        else:
            version_policy_str = "{ all { }}"

    # Use a different model name for the non-batching variant
    model_name = tu.get_model_name(
        "libtorch_nobatch" if max_batch == 0 else "libtorch", input_dtype,
        output0_dtype, output1_dtype)
    config_dir = models_dir + "/" + model_name
    config = '''
name: "{}"
platform: "pytorch_libtorch"
max_batch_size: {}
version_policy: {}
input [
  {{
    name: "INPUT__0"
    data_type: {}
    dims: [ {} ]
  }},
  {{
    name: "INPUT__1"
    data_type: {}
    dims: [ {} ]
  }}
]
output [
  {{
    name: "OUTPUT__0"
    data_type: {}
    dims: [ {} ]
    label_filename: "output0_labels.txt"
  }},
  {{
    name: "OUTPUT__1"
    data_type: {}
    dims: [ {} ]
  }}
]
'''.format(model_name, max_batch, version_policy_str,
           np_to_model_dtype(input_dtype), tu.shape_to_dims_str(input_shape),
           np_to_model_dtype(input_dtype), tu.shape_to_dims_str(input_shape),
           np_to_model_dtype(output0_dtype),
           tu.shape_to_dims_str(output0_shape),
           np_to_model_dtype(output1_dtype),
           tu.shape_to_dims_str(output1_shape))

    try:
        os.makedirs(config_dir)
    except OSError as ex:
        pass  # ignore existing dir

    with open(config_dir + "/config.pbtxt", "w") as cfile:
        cfile.write(config)

    with open(config_dir + "/output0_labels.txt", "w") as lfile:
        for l in range(output0_label_cnt):
            lfile.write("label" + str(l) + "\n")


def create_openvino_modelfile(models_dir,
                              max_batch,
                              model_version,
                              input_shape,
                              output0_shape,
                              output1_shape,
                              input_dtype,
                              output0_dtype,
                              output1_dtype,
                              swap=False):

    if not tu.validate_for_openvino_model(input_dtype, output0_dtype,
                                          output1_dtype, input_shape,
                                          output0_shape, output1_shape):
        return

    # Create the model
    model_name = tu.get_model_name(
        "openvino_nobatch" if max_batch == 0 else "openvino", input_dtype,
        output0_dtype, output1_dtype)
    model_version_dir = models_dir + "/" + model_name + "/" + str(model_version)

    batch_dim = () if max_batch == 0 else (max_batch,)
    in0 = ng.parameter(shape=batch_dim + input_shape, dtype=input_dtype, name="INPUT0")
    in1 = ng.parameter(shape=batch_dim + input_shape, dtype=input_dtype, name="INPUT1")

    r0 = ng.add(in0, in1) if not swap else ng.subtract(in0, in1)
    r1 = ng.subtract(in0, in1) if not swap else ng.add(in0, in1)

    result0 = ng.reshape(r0, batch_dim + output0_shape, special_zero=False)
    result1 = ng.reshape(r1, batch_dim + output1_shape, special_zero=False)

    op0 = ng.convert(result0, destination_type=output0_dtype, name="OUTPUT0")
    op1 = ng.convert(result1, destination_type=output1_dtype, name="OUTPUT1")

    function = ng.impl.Function([op0, op1], [in0, in1], model_name)
    ie_network = IENetwork(ng.impl.Function.to_capsule(function))

    # Batch size needs to be a positive integer value
    # if max_batch != 0:
    #     ie_network.batch_size = max_batch

    try:
        os.makedirs(model_version_dir)
    except OSError as ex:
        pass  # ignore existing dir

    ie_network.serialize(model_version_dir + "/model.xml",
                         model_version_dir + "/model.bin")


def create_openvino_modelconfig(models_dir, max_batch, model_version,
                                input_shape, output0_shape, output1_shape,
                                input_dtype, output0_dtype, output1_dtype,
                                output0_label_cnt, version_policy):

    if not tu.validate_for_openvino_model(input_dtype, output0_dtype,
                                          output1_dtype, input_shape,
                                          output0_shape, output1_shape):
        return

    # Unpack version policy
    version_policy_str = "{ latest { num_versions: 1 }}"
    if version_policy is not None:
        type, val = version_policy
        if type == 'latest':
            version_policy_str = "{{ latest {{ num_versions: {} }}}}".format(
                val)
        elif type == 'specific':
            version_policy_str = "{{ specific {{ versions: {} }}}}".format(val)
        else:
            version_policy_str = "{ all { }}"

    # Use a different model name for the non-batching variant
    model_name = tu.get_model_name(
        "openvino_nobatch" if max_batch == 0 else "openvino", input_dtype,
        output0_dtype, output1_dtype)
    config_dir = models_dir + "/" + model_name

    # platform is empty and backend is 'openvino' for openvino model
    config = '''
name: "{}"
backend: "openvino"
max_batch_size: {}
version_policy: {}
input [
  {{
    name: "INPUT0"
    data_type: {}
    dims: [ {} ]
  }},
  {{
    name: "INPUT1"
    data_type: {}
    dims: [ {} ]
  }}
]
output [
  {{
    name: "OUTPUT0"
    data_type: {}
    dims: [ {} ]
    label_filename: "output0_labels.txt"
   }},
  {{
    name: "OUTPUT1"
    data_type: {}
    dims: [ {} ]
  }}
]
'''.format(model_name, max_batch, version_policy_str,
           np_to_model_dtype(input_dtype), tu.shape_to_dims_str(input_shape),
           np_to_model_dtype(input_dtype), tu.shape_to_dims_str(input_shape),
           np_to_model_dtype(output0_dtype),
           tu.shape_to_dims_str(output0_shape),
           np_to_model_dtype(output1_dtype),
           tu.shape_to_dims_str(output1_shape))

    try:
        os.makedirs(config_dir)
    except OSError as ex:
        pass  # ignore existing dir

    with open(config_dir + "/config.pbtxt", "w") as cfile:
        cfile.write(config)

    with open(config_dir + "/output0_labels.txt", "w") as lfile:
        for l in range(output0_label_cnt):
            lfile.write("label" + str(l) + "\n")


def create_models(models_dir,
                  input_dtype,
                  output0_dtype,
                  output1_dtype,
                  input_shape,
                  output0_shape,
                  output1_shape,
                  output0_label_cnt,
                  version_policy=None):
    model_version = 1

    # Create two models, one that supports batching with a max-batch
    # of 8, and one that does not with a max-batch of 0
    if FLAGS.graphdef:
        # max-batch 8
        create_graphdef_modelconfig(models_dir, 8, model_version, input_shape,
                                    output0_shape, output1_shape, input_dtype,
                                    output0_dtype, output1_dtype,
                                    output0_label_cnt, version_policy)
        create_graphdef_modelfile(models_dir, 8, model_version, input_shape,
                                  output0_shape, output1_shape, input_dtype,
                                  output0_dtype, output1_dtype)
        # max-batch 0
        create_graphdef_modelconfig(models_dir, 0, model_version, input_shape,
                                    output0_shape, output1_shape, input_dtype,
                                    output0_dtype, output1_dtype,
                                    output0_label_cnt, version_policy)
        create_graphdef_modelfile(models_dir, 0, model_version, input_shape,
                                  output0_shape, output1_shape, input_dtype,
                                  output0_dtype, output1_dtype)

    if FLAGS.savedmodel:
        # max-batch 8
        create_savedmodel_modelconfig(models_dir, 8, model_version, input_shape,
                                      output0_shape, output1_shape, input_dtype,
                                      output0_dtype, output1_dtype,
                                      output0_label_cnt, version_policy)
        create_savedmodel_modelfile(models_dir, 8, model_version, input_shape,
                                    output0_shape, output1_shape, input_dtype,
                                    output0_dtype, output1_dtype)
        # max-batch 0
        create_savedmodel_modelconfig(models_dir, 0, model_version, input_shape,
                                      output0_shape, output1_shape, input_dtype,
                                      output0_dtype, output1_dtype,
                                      output0_label_cnt, version_policy)
        create_savedmodel_modelfile(models_dir, 0, model_version, input_shape,
                                    output0_shape, output1_shape, input_dtype,
                                    output0_dtype, output1_dtype)

    if FLAGS.tensorrt:
        # max-batch 8
        suffix = ()
        if input_dtype == np.int8 or output0_dtype == np.int8 or output1_dtype == np.int8:
            suffix = (1, 1)
        create_plan_modelconfig(models_dir, 8, model_version,
                                input_shape + suffix, output0_shape + suffix,
                                output1_shape + suffix, input_dtype,
                                output0_dtype, output1_dtype, output0_label_cnt,
                                version_policy)
        create_plan_modelfile(models_dir, 8, model_version,
                              input_shape + suffix, output0_shape + suffix,
                              output1_shape + suffix, input_dtype,
                              output0_dtype, output1_dtype)
        # max-batch 0
        create_plan_modelconfig(models_dir, 0, model_version,
                                input_shape + suffix, output0_shape + suffix,
                                output1_shape + suffix, input_dtype,
                                output0_dtype, output1_dtype, output0_label_cnt,
                                version_policy)
        create_plan_modelfile(models_dir, 0, model_version,
                              input_shape + suffix, output0_shape + suffix,
                              output1_shape + suffix, input_dtype,
                              output0_dtype, output1_dtype)

        if -1 in input_shape:
            # models for testing optimization profiles
            create_plan_modelconfig(models_dir,
                                    8,
                                    model_version,
                                    input_shape + suffix,
                                    output0_shape + suffix,
                                    output1_shape + suffix,
                                    input_dtype,
                                    output0_dtype,
                                    output1_dtype,
                                    output0_label_cnt,
                                    version_policy,
                                    min_dim=4,
                                    max_dim=32)
            create_plan_modelfile(models_dir,
                                  8,
                                  model_version,
                                  input_shape + suffix,
                                  output0_shape + suffix,
                                  output1_shape + suffix,
                                  input_dtype,
                                  output0_dtype,
                                  output1_dtype,
                                  min_dim=4,
                                  max_dim=32)

    if FLAGS.onnx:
        # max-batch 8
        create_onnx_modelconfig(models_dir, 8, model_version, input_shape,
                                output0_shape, output1_shape, input_dtype,
                                output0_dtype, output1_dtype, output0_label_cnt,
                                version_policy)
        create_onnx_modelfile(models_dir, 8, model_version, input_shape,
                              output0_shape, output1_shape, input_dtype,
                              output0_dtype, output1_dtype)
        # max-batch 0
        create_onnx_modelconfig(models_dir, 0, model_version, input_shape,
                                output0_shape, output1_shape, input_dtype,
                                output0_dtype, output1_dtype, output0_label_cnt,
                                version_policy)
        create_onnx_modelfile(models_dir, 0, model_version, input_shape,
                              output0_shape, output1_shape, input_dtype,
                              output0_dtype, output1_dtype)

    if FLAGS.libtorch:
        # max-batch 8
        create_libtorch_modelconfig(models_dir, 8, model_version, input_shape,
                                    output0_shape, output1_shape, input_dtype,
                                    output0_dtype, output1_dtype,
                                    output0_label_cnt, version_policy)
        create_libtorch_modelfile(models_dir, 8, model_version, input_shape,
                                  output0_shape, output1_shape, input_dtype,
                                  output0_dtype, output1_dtype)
        # max-batch 0
        create_libtorch_modelconfig(models_dir, 0, model_version, input_shape,
                                    output0_shape, output1_shape, input_dtype,
                                    output0_dtype, output1_dtype,
                                    output0_label_cnt, version_policy)
        create_libtorch_modelfile(models_dir, 0, model_version, input_shape,
                                  output0_shape, output1_shape, input_dtype,
                                  output0_dtype, output1_dtype)

    if FLAGS.openvino:
        # max-batch 8
        create_openvino_modelconfig(models_dir, 8, model_version, input_shape,
                                    output0_shape, output1_shape, input_dtype,
                                    output0_dtype, output1_dtype,
                                    output0_label_cnt, version_policy)
        create_openvino_modelfile(models_dir, 8, model_version, input_shape,
                                  output0_shape, output1_shape, input_dtype,
                                  output0_dtype, output1_dtype)
        # max-batch 0
        create_openvino_modelconfig(models_dir, 0, model_version, input_shape,
                                    output0_shape, output1_shape, input_dtype,
                                    output0_dtype, output1_dtype,
                                    output0_label_cnt, version_policy)
        create_openvino_modelfile(models_dir, 0, model_version, input_shape,
                                  output0_shape, output1_shape, input_dtype,
                                  output0_dtype, output1_dtype)

    if FLAGS.ensemble:
        for pair in emu.platform_types_and_validation():
            if not pair[1](input_dtype, output0_dtype, output1_dtype,
                           input_shape, output0_shape, output1_shape):
                continue

            config_input_shape = input_shape
            config_output0_shape = output0_shape
            config_output1_shape = output1_shape
            if pair[0] == "plan":
                if len(input_shape) == 1 and input_dtype == np.int8:
                    config_input_shape = (input_shape[0], 1, 1)
                if len(output0_shape) == 1 and output0_dtype == np.int8:
                    config_output0_shape = (output0_shape[0], 1, 1)
                if len(output1_shape) == 1 and output1_dtype == np.int8:
                    config_output1_shape = (output1_shape[0], 1, 1)

            # max-batch 8
            emu.create_ensemble_modelconfig(pair[0], models_dir, 8,
                                            model_version, config_input_shape,
                                            config_output0_shape,
                                            config_output1_shape, input_dtype,
                                            output0_dtype, output1_dtype,
                                            output0_label_cnt, version_policy)
            emu.create_ensemble_modelfile(pair[0], models_dir, 8, model_version,
                                          config_input_shape,
                                          config_output0_shape,
                                          config_output1_shape, input_dtype,
                                          output0_dtype, output1_dtype)
            # max-batch 0
            emu.create_ensemble_modelconfig(pair[0], models_dir, 0,
                                            model_version, config_input_shape,
                                            config_output0_shape,
                                            config_output1_shape, input_dtype,
                                            output0_dtype, output1_dtype,
                                            output0_label_cnt, version_policy)
            emu.create_ensemble_modelfile(pair[0], models_dir, 0, model_version,
                                          config_input_shape,
                                          config_output0_shape,
                                          config_output1_shape, input_dtype,
                                          output0_dtype, output1_dtype)


def create_fixed_models(models_dir,
                        input_dtype,
                        output0_dtype,
                        output1_dtype,
                        version_policy=None):
    input_size = 16
    create_models(models_dir, input_dtype, output0_dtype, output1_dtype,
                  (input_size,), (input_size,), (input_size,), input_size,
                  version_policy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_dir',
                        type=str,
                        required=True,
                        help='Top-level model directory')
    parser.add_argument('--graphdef',
                        required=False,
                        action='store_true',
                        help='Generate GraphDef models')
    parser.add_argument('--savedmodel',
                        required=False,
                        action='store_true',
                        help='Generate SavedModel models')
    parser.add_argument('--tensorrt',
                        required=False,
                        action='store_true',
                        help='Generate TensorRT PLAN models')
    parser.add_argument('--onnx',
                        required=False,
                        action='store_true',
                        help='Generate Onnx Runtime Onnx models')
    parser.add_argument(
        '--onnx_opset',
        type=int,
        required=False,
        default=0,
        help='Opset used for Onnx models. Default is to use ONNXRT default')
    parser.add_argument('--libtorch',
                        required=False,
                        action='store_true',
                        help='Generate Pytorch LibTorch models')
    parser.add_argument('--openvino',
                        required=False,
                        action='store_true',
                        help='Generate Openvino models')
    parser.add_argument('--variable',
                        required=False,
                        action='store_true',
                        help='Used variable-shape tensors for input/output')
    parser.add_argument('--ensemble',
                        required=False,
                        action='store_true',
                        help='Generate ensemble models against the models' +
                        ' in all platforms. Note that the models generated' +
                        ' are not completed.')
    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.graphdef or FLAGS.savedmodel:
        import tensorflow as tf
        from tensorflow.python.framework import graph_io, graph_util
    if FLAGS.tensorrt:
        import tensorrt as trt
    if FLAGS.onnx:
        import onnx
    if FLAGS.libtorch:
        import torch
        from torch import nn
    if FLAGS.openvino:
        from openvino.inference_engine import IECore, IENetwork
        import ngraph as ng

    import test_util as tu

    # Tests with models that accept fixed-shape input/output tensors
    if not FLAGS.variable:
        create_fixed_models(FLAGS.models_dir, np.int8, np.int8, np.int8,
                            ('latest', 1))
        create_fixed_models(FLAGS.models_dir, np.int16, np.int16, np.int16,
                            ('latest', 2))
        create_fixed_models(FLAGS.models_dir, np.int32, np.int32, np.int32,
                            ('all', None))
        create_fixed_models(FLAGS.models_dir, np.int64, np.int64, np.int64)
        create_fixed_models(FLAGS.models_dir, np.float16, np.float16,
                            np.float16, ('specific', [
                                1,
                            ]))
        create_fixed_models(FLAGS.models_dir, np.float32, np.float32,
                            np.float32, ('specific', [1, 3]))
        create_fixed_models(FLAGS.models_dir, np.float16, np.float32,
                            np.float32)
        create_fixed_models(FLAGS.models_dir, np.int32, np.int8, np.int8)
        create_fixed_models(FLAGS.models_dir, np.int8, np.int32, np.int32)
        create_fixed_models(FLAGS.models_dir, np.int32, np.int8, np.int16)
        create_fixed_models(FLAGS.models_dir, np.int32, np.float32, np.float32)
        create_fixed_models(FLAGS.models_dir, np.float32, np.int32, np.int32)
        create_fixed_models(FLAGS.models_dir, np.int32, np.float16, np.int16)

        create_fixed_models(FLAGS.models_dir, np_dtype_string, np.int32,
                            np.int32)
        create_fixed_models(FLAGS.models_dir, np_dtype_string, np_dtype_string,
                            np_dtype_string)
        create_fixed_models(FLAGS.models_dir, np_dtype_string, np.int32,
                            np_dtype_string)
        create_fixed_models(FLAGS.models_dir, np_dtype_string, np_dtype_string,
                            np.int32)
        create_fixed_models(FLAGS.models_dir, np.int32, np_dtype_string,
                            np_dtype_string)
        create_fixed_models(FLAGS.models_dir, np.int32, np.int32,
                            np_dtype_string)
        create_fixed_models(FLAGS.models_dir, np.int32, np_dtype_string,
                            np.int32)

        # Make multiple versions of some models for version testing
        # (they use different version policies when created above)
        if FLAGS.graphdef:
            for vt in [np.float16, np.float32, np.int8, np.int16, np.int32]:
                create_graphdef_modelfile(FLAGS.models_dir,
                                          8,
                                          2, (16,), (16,), (16,),
                                          vt,
                                          vt,
                                          vt,
                                          swap=True)
                create_graphdef_modelfile(FLAGS.models_dir,
                                          8,
                                          3, (16,), (16,), (16,),
                                          vt,
                                          vt,
                                          vt,
                                          swap=True)
                create_graphdef_modelfile(FLAGS.models_dir,
                                          0,
                                          2, (16,), (16,), (16,),
                                          vt,
                                          vt,
                                          vt,
                                          swap=True)
                create_graphdef_modelfile(FLAGS.models_dir,
                                          0,
                                          3, (16,), (16,), (16,),
                                          vt,
                                          vt,
                                          vt,
                                          swap=True)

        if FLAGS.savedmodel:
            for vt in [np.float16, np.float32, np.int8, np.int16, np.int32]:
                create_savedmodel_modelfile(FLAGS.models_dir,
                                            8,
                                            2, (16,), (16,), (16,),
                                            vt,
                                            vt,
                                            vt,
                                            swap=True)
                create_savedmodel_modelfile(FLAGS.models_dir,
                                            8,
                                            3, (16,), (16,), (16,),
                                            vt,
                                            vt,
                                            vt,
                                            swap=True)
                create_savedmodel_modelfile(FLAGS.models_dir,
                                            0,
                                            2, (16,), (16,), (16,),
                                            vt,
                                            vt,
                                            vt,
                                            swap=True)
                create_savedmodel_modelfile(FLAGS.models_dir,
                                            0,
                                            3, (16,), (16,), (16,),
                                            vt,
                                            vt,
                                            vt,
                                            swap=True)

        if FLAGS.tensorrt:
            for vt in [np.float32, np.float16, np.int32]:
                create_plan_modelfile(FLAGS.models_dir,
                                      8,
                                      2, (16,), (16,), (16,),
                                      vt,
                                      vt,
                                      vt,
                                      swap=True)
                create_plan_modelfile(FLAGS.models_dir,
                                      8,
                                      3, (16,), (16,), (16,),
                                      vt,
                                      vt,
                                      vt,
                                      swap=True)
                create_plan_modelfile(FLAGS.models_dir,
                                      0,
                                      2, (16,), (16,), (16,),
                                      vt,
                                      vt,
                                      vt,
                                      swap=True)
                create_plan_modelfile(FLAGS.models_dir,
                                      0,
                                      3, (16,), (16,), (16,),
                                      vt,
                                      vt,
                                      vt,
                                      swap=True)

            vt = np.int8
            #handle INT8 separately as it doesn't allow 1d tensors
            create_plan_modelfile(FLAGS.models_dir,
                                  8,
                                  2, (16, 1, 1), (16, 1, 1), (16, 1, 1),
                                  vt,
                                  vt,
                                  vt,
                                  swap=True)
            create_plan_modelfile(FLAGS.models_dir,
                                  8,
                                  3, (16, 1, 1), (16, 1, 1), (16, 1, 1),
                                  vt,
                                  vt,
                                  vt,
                                  swap=True)
            create_plan_modelfile(FLAGS.models_dir,
                                  0,
                                  2, (16, 1, 1), (16, 1, 1), (16, 1, 1),
                                  vt,
                                  vt,
                                  vt,
                                  swap=True)
            create_plan_modelfile(FLAGS.models_dir,
                                  0,
                                  3, (16, 1, 1), (16, 1, 1), (16, 1, 1),
                                  vt,
                                  vt,
                                  vt,
                                  swap=True)

        if FLAGS.onnx:
            for vt in [np.float16, np.float32, np.int8, np.int16, np.int32]:
                create_onnx_modelfile(FLAGS.models_dir,
                                      8,
                                      2, (16,), (16,), (16,),
                                      vt,
                                      vt,
                                      vt,
                                      swap=True)
                create_onnx_modelfile(FLAGS.models_dir,
                                      8,
                                      3, (16,), (16,), (16,),
                                      vt,
                                      vt,
                                      vt,
                                      swap=True)
                create_onnx_modelfile(FLAGS.models_dir,
                                      0,
                                      2, (16,), (16,), (16,),
                                      vt,
                                      vt,
                                      vt,
                                      swap=True)
                create_onnx_modelfile(FLAGS.models_dir,
                                      0,
                                      3, (16,), (16,), (16,),
                                      vt,
                                      vt,
                                      vt,
                                      swap=True)
        if FLAGS.libtorch:
            for vt in [np.float32, np.int32, np.int16, np.int8]:
                create_libtorch_modelfile(FLAGS.models_dir,
                                          8,
                                          2, (16,), (16,), (16,),
                                          vt,
                                          vt,
                                          vt,
                                          swap=True)
                create_libtorch_modelfile(FLAGS.models_dir,
                                          8,
                                          3, (16,), (16,), (16,),
                                          vt,
                                          vt,
                                          vt,
                                          swap=True)
                create_libtorch_modelfile(FLAGS.models_dir,
                                          0,
                                          2, (16,), (16,), (16,),
                                          vt,
                                          vt,
                                          vt,
                                          swap=True)
                create_libtorch_modelfile(FLAGS.models_dir,
                                          0,
                                          3, (16,), (16,), (16,),
                                          vt,
                                          vt,
                                          vt,
                                          swap=True)
        if FLAGS.openvino:
            for vt in [np.float16, np.float32, np.int8, np.int16, np.int32]:
                create_openvino_modelfile(FLAGS.models_dir,
                                          8,
                                          2, (16,), (16,), (16,),
                                          vt,
                                          vt,
                                          vt,
                                          swap=True)
                create_openvino_modelfile(FLAGS.models_dir,
                                          8,
                                          3, (16,), (16,), (16,),
                                          vt,
                                          vt,
                                          vt,
                                          swap=True)
                create_openvino_modelfile(FLAGS.models_dir,
                                          0,
                                          2, (16,), (16,), (16,),
                                          vt,
                                          vt,
                                          vt,
                                          swap=True)
                create_openvino_modelfile(FLAGS.models_dir,
                                          0,
                                          3, (16,), (16,), (16,),
                                          vt,
                                          vt,
                                          vt,
                                          swap=True)

        if FLAGS.ensemble:
            for pair in emu.platform_types_and_validation():
                for vt in [np.float16, np.float32, np.int8, np.int16, np.int32]:
                    shape = (16, 1, 1) if (pair[0] == "plan" and
                                           vt == np.int8) else (16,)
                    if not pair[1](vt, vt, vt, shape, shape, shape):
                        continue
                    emu.create_ensemble_modelfile(pair[0],
                                                  FLAGS.models_dir,
                                                  8,
                                                  2,
                                                  shape,
                                                  shape,
                                                  shape,
                                                  vt,
                                                  vt,
                                                  vt,
                                                  swap=True)
                    emu.create_ensemble_modelfile(pair[0],
                                                  FLAGS.models_dir,
                                                  8,
                                                  3,
                                                  shape,
                                                  shape,
                                                  shape,
                                                  vt,
                                                  vt,
                                                  vt,
                                                  swap=True)
                    emu.create_ensemble_modelfile(pair[0],
                                                  FLAGS.models_dir,
                                                  0,
                                                  2,
                                                  shape,
                                                  shape,
                                                  shape,
                                                  vt,
                                                  vt,
                                                  vt,
                                                  swap=True)
                    emu.create_ensemble_modelfile(pair[0],
                                                  FLAGS.models_dir,
                                                  0,
                                                  3,
                                                  shape,
                                                  shape,
                                                  shape,
                                                  vt,
                                                  vt,
                                                  vt,
                                                  swap=True)

    # Tests with models that accept variable-shape input/output tensors
    if FLAGS.variable:
        create_models(FLAGS.models_dir, np.float32, np.float32, np.float32,
                      (-1,), (-1,), (-1,), 16)
        create_models(FLAGS.models_dir, np.float32, np.int32, np.int32,
                      (-1, -1), (-1, -1), (-1, -1), 16)
        create_models(FLAGS.models_dir, np.float32, np.int64, np.int64, (8, -1),
                      (8, -1), (8, -1), 32)
        create_models(FLAGS.models_dir, np.float32, np.int32, np.int64,
                      (-1, 8, -1), (-1, 8, -1), (-1, 8, -1), 32)
        create_models(FLAGS.models_dir, np.float32, np.float32, np.int32, (-1,),
                      (-1,), (-1,), 16)
        create_models(FLAGS.models_dir, np.int32, np.int32, np.int32, (-1, -1),
                      (-1, -1), (-1, -1), 16)
        create_models(FLAGS.models_dir, np.int32, np.int32, np.float32,
                      (-1, 8, -1), (-1, 8, -1), (-1, 8, -1), 32)

        create_models(FLAGS.models_dir, np_dtype_string, np_dtype_string,
                      np_dtype_string, (-1,), (-1,), (-1,), 16)
        create_models(FLAGS.models_dir, np_dtype_string, np.int32, np.int32,
                      (-1, -1), (-1, -1), (-1, -1), 16)
        create_models(FLAGS.models_dir, np_dtype_string, np_dtype_string,
                      np.int32, (8, -1), (8, -1), (8, -1), 32)
        create_models(FLAGS.models_dir, np_dtype_string, np.int32,
                      np_dtype_string, (-1, 8, -1), (-1, 8, -1), (-1, 8, -1),
                      32)

    if FLAGS.ensemble:
        # Create utility models used in ensemble
        # nop (only creates model config, should add model file before use)
        model_dtypes = ["TYPE_BOOL", "TYPE_STRING"]
        for s in [8, 16, 32, 64]:
            for t in ["INT", "UINT", "FP"]:
                if t == "FP" and s == 8:
                    continue
                model_dtypes.append("TYPE_{}{}".format(t, s))

        for model_dtype in model_dtypes:
            # Use variable size to handle all shape. Note: piping variable size output
            # to fixed size model is not safe but doable
            for model_shape in [(-1,), (-1, -1), (-1, -1, -1)]:
                emu.create_nop_modelconfig(FLAGS.models_dir, model_shape,
                                           model_dtype)
