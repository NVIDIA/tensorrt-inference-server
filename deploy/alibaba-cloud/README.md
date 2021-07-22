# Deploy Triton Inference Server on PAI-EAS 
* Table Of Contents
   - [Description](https://yuque.alibaba-inc.com/pai/blade/mtptqc#Description)
   - [Prerequisites](https://yuque.alibaba-inc.com/pai/blade/mtptqc#Prerequisites)
   - [Demo Instruction](https://yuque.alibaba-inc.com/pai/blade/mtptqc#31bb94ef)
   - [Additional Resources](https://yuque.alibaba-inc.com/pai/blade/mtptqc#89d5e680)
   - [Known Issues](https://yuque.alibaba-inc.com/pai/blade/mtptqc#558ab0be)

# Description
This repository contains infomation about how to deploy NVIDIA Triton Inference Server in  EAS(Elastic Algorithm Service) of Alibaba-Cloud.
- EAS providing a simple way for deep learning developers to deploy their models in Alibaba Cloud.
- Using **Triton Processor** is the recommended way on EAS to deploy Triton Inference Server. Users can simply deploy a Triton Server by preparing models and creating a EAS service by setting processor type to `triton`.
- Models should be uploaded to Alibaba Cloud's OSS(Object Storage Service). User's model repository in OSS will be mounted onto local path visible to Triton Server.
- This documentation uses Triton's own example models for demo. The tensorflow inception model can be downloaded by the `fetch_models.sh` script.

# Prerequisites
- You should register an Alibaba Cloud Account, and being able to use EAS by [eascmd](https://help.aliyun.com/document_detail/111031.html?spm=a2c4g.11186623.6.752.42356f46FN5fU1), which is a command line tool to create stop or scale services on EAS.
- Before creating an EAS service, you should buy dedicated resource groups(CPU or GPU) on EAS following this [document](https://www.alibabacloud.com/help/doc-detail/120122.htm).
- Make sure you can use OSS(Object Storage Service), the models should be uploaded into your own OSS bucket.

# Demo Instruction
## Prepare a model repo directory in OSS
Download the tensorflow inception model via [fetch_model.sh](https://github.com/triton-inference-server/server/blob/main/docs/examples/fetch_models.sh). Then using [ossutil](https://help.aliyun.com/document_detail/50452.html?spm=a2c4g.11186623.6.833.26d66d51dPEytI) , which is a command line tool to use OSS, to upload the model to a certain OSS dir as you want.

```
./ossutil cp inception_graphdef/ oss://triton-model-repo/models
```
## Create Triton Service with json config by eascmd
The following is the json we use when creating a Triton Server on EAS.
```
{
  "name": "<your triton service name>",                          
  "processor": "triton",
  "processor_params": [
    "--model-repository=oss://triton-model-repo/models", 
    "--allow-grpc=true", 
    "--allow-http=true"
  ],
  "metadata": {
    "instance": 1,
    "cpu": 4,
    "gpu": 1,
    "memory": 10000,
    "resource": "<your resource id>",
    "rpc.keepalive": 3000
  }
}
```
Only processor and processor_params should be different from a normal EAS service.
|params|details|
|--------|-------|
|processor|Name should be **triton** to use Triton on EAS|
|processor_params|List of strings, every element is a param for tritonserver |

```
./eascmd create triton.config
[RequestId]: AECDB6A4-CB69-4688-AA35-BA1E020C39E6
+-------------------+------------------------------------------------------------------------------------------------+
| Internet Endpoint | http://1271520832287160.cn-shanghai.pai-eas.aliyuncs.com/api/predict/test_triton_processor     |
| Intranet Endpoint | http://1271520832287160.vpc.cn-shanghai.pai-eas.aliyuncs.com/api/predict/test_triton_processor |
|             Token | MmY3M2ExZGYwYjZiMTQ5YTRmZWE3MDAzNWM1ZTBiOWQ3MGYxZGNkZQ==                                       |
+-------------------+------------------------------------------------------------------------------------------------+
[OK] Service is now deploying
[OK] Successfully synchronized resources
[OK] Waiting [Total: 1, Pending: 1, Running: 0]
[OK] Waiting [Total: 1, Pending: 1, Running: 0]
[OK] Running [Total: 1, Pending: 0, Running: 1]
[OK] Service is running
```
## Query Triton service by python client
### Install triton's python client
```
pip3 install nvidia-pyindex
pip install tritonclient[all]
```
### A demo to query inception model
```
import numpy as np
import time
from PIL import Image

import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

URL = "<servcice url>"
HEADERS = {"Authorization": "<service token>"}
input_img = httpclient.InferInput("input", [1, 299, 299, 3], "FP32")
img = Image.open('./cat.png').resize((299, 299))
img = np.asarray(img).astype('float32') / 255.0
input_img.set_data_from_numpy(img.reshape([1, 299, 299, 3]), binary_data=True)

output = httpclient.InferRequestedOutput(
    "InceptionV3/Predictions/Softmax", binary_data=True
)
triton_client = httpclient.InferenceServerClient(url=URL, verbose=False)

start = time.time()
for i in range(10):
    results = triton_client.infer(
        "inception_graphdef", inputs=[input_img], outputs=[output], headers=HEADERS
    )
    res_body = results.get_response()
    elapsed_ms = (time.time() - start) * 1000
    if i == 0:
        print("model name: ", res_body["model_name"])
        print("model version: ", res_body["model_version"])
        print("output name: ", res_body["outputs"][0]["name"])
        print("output shape: ", res_body["outputs"][0]["shape"])
    print("[{}] Avg rt(ms): {:.2f}".format(i, elapsed_ms))
    start = time.time()
```
You will get the following result by running the python script:
```
[0] Avg rt(ms): 86.05
[1] Avg rt(ms): 52.35
[2] Avg rt(ms): 50.56
[3] Avg rt(ms): 43.45
[4] Avg rt(ms): 41.19
[5] Avg rt(ms): 40.55
[6] Avg rt(ms): 37.24
[7] Avg rt(ms): 37.16
[8] Avg rt(ms): 36.68
[9] Avg rt(ms): 34.24
[10] Avg rt(ms): 34.27
```
# Additional Resources
See the following resources to learn more about how to use Alibaba Cloud's OSS orEAS.
- [Alibaba Cloud OSS's Document](https://help.aliyun.com/product/31815.html?spm=a2c4g.11186623.6.540.3c0f62e7q3jw8b)


# Known Issues
- [Binary Tensor Data Extension](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_binary_data.md) is not fully supported yet, users want to use service with binary extension supported, it is only available in cn-shanghai region of PAI-EAS.
- Currently only HTTP/1 is supported, hence gRPC cannot be used when query Triton servers on EAS. HTP/2 will be officially supported in a short time.
- Users should not mount a whole OSS bucket when launching Triton processor, but an arbitrarily deep sub-directory in bucket. Otherwise the mounted path will no be as expected.
- Not all of Triton Server parameters are be supported on EAS, the following params are the ones supported now.
```
log-verbose
log-info
log-warning
log-error
exit-on-error
strict-model-config
strict-readiness
allow-http
http-thread-count
pinned-memory-pool-byte-size
cuda-memory-pool-byte-size
min-supported-compute-capability
buffer-manager-thread-count
backend-config
```
