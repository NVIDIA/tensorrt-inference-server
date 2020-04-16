// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "src/clients/c++/experimental_api_v2/library/http_client.h"

#include <curl/curl.h>
#include <cstdint>
#include <iostream>
#include <queue>

namespace nvidia { namespace inferenceserver { namespace client {

namespace {

//==============================================================================

// Global initialization for libcurl. Libcurl requires global
// initialization before any other threads are created and before any
// curl methods are used. The curl_global static object is used to
// perform this initialization.
class CurlGlobal {
 public:
  CurlGlobal();
  ~CurlGlobal();

  const Error& Status() const { return err_; }

 private:
  Error err_;
};

CurlGlobal::CurlGlobal() : err_(Error::Success)
{
  if (curl_global_init(CURL_GLOBAL_ALL) != 0) {
    err_ = Error("global initialization failed");
  }
}

CurlGlobal::~CurlGlobal()
{
  curl_global_cleanup();
}

static CurlGlobal curl_global;

std::string
GetQueryString(const Headers& query_params)
{
  std::string query_string;
  bool first = true;
  for (const auto& pr : query_params) {
    if (first) {
      first = false;
    } else {
      query_string += "&";
    }
    query_string += pr.first + "=" + pr.second;
  }
  return query_string;
}

}  // namespace

//==============================================================================

std::string
GetJsonText(const rapidjson::Document& json_dom)
{
  rapidjson::StringBuffer buffer;
  rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
  json_dom.Accept(writer);
  return buffer.GetString();
}

//==============================================================================

class HttpInferRequest : public InferRequest {
 public:
  HttpInferRequest();
  ~HttpInferRequest();

  // Initialize the request for HTTP transfer. */
  Error InitializeRequest(rapidjson::Document& response_json);

  // Adds the input data to be delivered to the server
  Error AddInput(uint8_t* buf, size_t byte_size);

  // Copy into 'buf' up to 'size' bytes of input data. Return the
  // actual amount copied in 'input_bytes'.
  Error GetNextInput(uint8_t* buf, size_t size, size_t* input_bytes);

 private:
  friend class InferenceServerHttpClient;

  // Pointer to easy handle that is processing the request
  CURL* easy_handle_;

  // Pointer to the list of the HTTP request header, keep it such that it will
  // be valid during the transfer and can be freed once transfer is completed.
  struct curl_slist* header_list_;

  // Status code for the HTTP request.
  CURLcode http_status_;

  size_t total_input_byte_size_;

  rapidjson::StringBuffer request_json_;

  // Buffer that accumulates the serialized response at the
  // end of the body.
  std::unique_ptr<std::string> infer_response_buffer_;

  // The pointers to the input data.
  std::queue<std::pair<uint8_t*, size_t>> data_buffers_;

  size_t response_json_size_;
};


HttpInferRequest::HttpInferRequest()
    : easy_handle_(curl_easy_init()), header_list_(nullptr),
      total_input_byte_size_(0), response_json_size_(0)
{
}

HttpInferRequest::~HttpInferRequest()
{
  if (header_list_ != nullptr) {
    curl_slist_free_all(header_list_);
    header_list_ = nullptr;
  }

  if (easy_handle_ != nullptr) {
    curl_easy_cleanup(easy_handle_);
  }
}

Error
HttpInferRequest::InitializeRequest(rapidjson::Document& request_json)
{
  data_buffers_ = {};
  total_input_byte_size_ = 0;

  request_json_.Clear();
  rapidjson::Writer<rapidjson::StringBuffer> writer(request_json_);
  request_json.Accept(writer);

  // Add the buffer holding the json to be delivered first
  AddInput((uint8_t*)request_json_.GetString(), request_json_.GetSize());

  // Prepare buffer to record the response
  infer_response_buffer_.reset(new std::string());

  return Error::Success;
}

Error
HttpInferRequest::AddInput(uint8_t* buf, size_t byte_size)
{
  data_buffers_.push(std::pair<uint8_t*, size_t>(buf, byte_size));
  total_input_byte_size_ += byte_size;
  return Error::Success;
}

Error
HttpInferRequest::GetNextInput(uint8_t* buf, size_t size, size_t* input_bytes)
{
  *input_bytes = 0;

  if (data_buffers_.empty()) {
    return Error::Success;
  }

  while (!data_buffers_.empty() && size > 0) {
    const size_t csz = std::min(data_buffers_.front().second, size);
    if (csz > 0) {
      const uint8_t* input_ptr = data_buffers_.front().first;
      std::copy(input_ptr, input_ptr + csz, buf);
      size -= csz;
      buf += csz;
      *input_bytes += csz;


      data_buffers_.front().first += csz;
      data_buffers_.front().second -= csz;
      if (data_buffers_.front().second == 0) {
        data_buffers_.pop();
      }
    }
  }

  // Set end timestamp if all inputs have been sent.
  if (data_buffers_.empty()) {
    Timer().CaptureTimestamp(RequestTimers::Kind::SEND_END);
  }

  return Error::Success;
}

//==============================================================================

Error
InferenceServerHttpClient::Create(
    std::unique_ptr<InferenceServerHttpClient>* client,
    const std::string& server_url, bool verbose)
{
  client->reset(new InferenceServerHttpClient(server_url, verbose));
  client->get()->sync_request_.reset(
      static_cast<InferRequest*>(new HttpInferRequest()));
  return Error::Success;
}

Error
InferenceServerHttpClient::IsServerLive(
    bool* live, const Headers& headers, const Parameters& query_params)
{
  Error err;

  std::string request_uri(url_ + "/v2/health/live");

  long http_code;
  rapidjson::Document response;
  err = Get(request_uri, headers, query_params, &response, &http_code);

  *live = (http_code == 200) ? true : false;

  return err;
}

Error
InferenceServerHttpClient::IsServerReady(
    bool* ready, const Headers& headers, const Parameters& query_params)
{
  Error err;

  std::string request_uri(url_ + "/v2/health/live");

  long http_code;
  rapidjson::Document response;
  err = Get(request_uri, headers, query_params, &response, &http_code);

  *ready = (http_code == 200) ? true : false;

  return err;
}

Error
InferenceServerHttpClient::IsModelReady(
    bool* ready, const std::string& model_name,
    const std::string& model_version, const Headers& headers,
    const Parameters& query_params)
{
  Error err;

  std::string request_uri(url_ + "/v2/models/" + model_name);
  if (!model_version.empty()) {
    request_uri = request_uri + "/versions/" + model_version;
  }
  request_uri = request_uri + "/ready";

  long http_code;
  rapidjson::Document response;
  err = Get(request_uri, headers, query_params, &response, &http_code);

  *ready = (http_code == 200) ? true : false;

  return err;
}


Error
InferenceServerHttpClient::GetServerMetadata(
    rapidjson::Document* server_metadata, const Headers& headers,
    const Parameters& query_params)
{
  Error err;

  std::string request_uri(url_ + "/v2");

  long http_code;
  err = Get(request_uri, headers, query_params, server_metadata, &http_code);
  if ((http_code != 200) && err.IsOk()) {
    return Error(
        "[INTERNAL] Request failed with missing error message in response");
  }
  return err;
}


Error
InferenceServerHttpClient::GetModelMetadata(
    rapidjson::Document* model_metadata, const std::string& model_name,
    const std::string& model_version, const Headers& headers,
    const Parameters& query_params)
{
  Error err;

  std::string request_uri(url_ + "/v2/models/" + model_name);
  if (!model_version.empty()) {
    request_uri = request_uri + "/versions/" + model_version;
  }

  long http_code;
  err = Get(request_uri, headers, query_params, model_metadata, &http_code);
  if ((http_code != 200) && err.IsOk()) {
    return Error(
        "[INTERNAL] Request failed with missing error message in response");
  }
  return err;
}


Error
InferenceServerHttpClient::GetModelConfig(
    rapidjson::Document* model_config, const std::string& model_name,
    const std::string& model_version, const Headers& headers,
    const Parameters& query_params)
{
  Error err;

  std::string request_uri(url_ + "/v2/models/" + model_name);
  if (!model_version.empty()) {
    request_uri = request_uri + "/versions/" + model_version;
  }
  request_uri = request_uri + "/config";

  long http_code;
  err = Get(request_uri, headers, query_params, model_config, &http_code);
  if ((http_code != 200) && err.IsOk()) {
    return Error(
        "[INTERNAL] Request failed with missing error message in response");
  }
  return err;
}

Error
InferenceServerHttpClient::Infer(
    InferResult** result, const InferOptions& options,
    const std::vector<InferInput*>& inputs,
    const std::vector<const InferRequestedOutput*>& outputs,
    const Headers& headers, const Parameters& query_params)
{
  Error err;

  std::string request_uri(url_ + "/v2/models/" + options.model_name_);
  if (!options.model_version_.empty()) {
    request_uri = request_uri + "/versions/" + options.model_version_;
  }
  request_uri = request_uri + "/infer";

  std::shared_ptr<HttpInferRequest> sync_request =
      std::static_pointer_cast<HttpInferRequest>(sync_request_);

  sync_request->Timer().Reset();
  sync_request->Timer().CaptureTimestamp(RequestTimers::Kind::REQUEST_START);

  if (!curl_global.Status().IsOk()) {
    return curl_global.Status();
  }

  err = PreRunProcessing(
      request_uri, options, inputs, outputs, headers, query_params,
      sync_request_);
  if (!err.IsOk()) {
    return err;
  }

  sync_request->Timer().CaptureTimestamp(RequestTimers::Kind::SEND_START);

  // During this call SEND_END, RECV_START and RECV_END will be set.
  sync_request->http_status_ = curl_easy_perform(sync_request->easy_handle_);

  InferResultHttp::Create(
      result, std::move(sync_request->infer_response_buffer_),
      sync_request->response_json_size_);

  sync_request->Timer().CaptureTimestamp(RequestTimers::Kind::REQUEST_END);

  err = UpdateInferStat(sync_request->Timer());
  if (!err.IsOk()) {
    std::cerr << "Failed to update context stat: " << err << std::endl;
  }

  err = reinterpret_cast<InferResultHttp*>(*result)->RequestStatus();

  return err;
}

InferenceServerHttpClient::InferenceServerHttpClient(
    const std::string& url, bool verbose)
    : url_(url), verbose_(verbose)
{
}


size_t
InferenceServerHttpClient::InferRequestProvider(
    void* contents, size_t size, size_t nmemb, void* userp)
{
  HttpInferRequest* request = reinterpret_cast<HttpInferRequest*>(userp);

  size_t input_bytes = 0;
  Error err = request->GetNextInput(
      reinterpret_cast<uint8_t*>(contents), size * nmemb, &input_bytes);
  if (!err.IsOk()) {
    std::cerr << "RequestProvider: " << err << std::endl;
    return CURL_READFUNC_ABORT;
  }

  return input_bytes;
}

size_t
InferenceServerHttpClient::InferResponseHeaderHandler(
    void* contents, size_t size, size_t nmemb, void* userp)
{
  HttpInferRequest* request = reinterpret_cast<HttpInferRequest*>(userp);

  char* buf = reinterpret_cast<char*>(contents);
  size_t byte_size = size * nmemb;

  size_t idx = strlen(kInferHeaderContentLengthHTTPHeader);
  if ((idx < byte_size) &&
      !strncasecmp(buf, kInferHeaderContentLengthHTTPHeader, idx)) {
    while ((idx < byte_size) && (buf[idx] != ':')) {
      ++idx;
    }

    if (idx < byte_size) {
      std::string hdr(buf + idx + 1, byte_size - idx - 1);
      request->response_json_size_ = std::stoi(hdr);
    }
  }

  return byte_size;
}

size_t
InferenceServerHttpClient::InferResponseHandler(
    void* contents, size_t size, size_t nmemb, void* userp)
{
  HttpInferRequest* request = reinterpret_cast<HttpInferRequest*>(userp);

  if (request->Timer().Timestamp(RequestTimers::Kind::RECV_START) == 0) {
    request->Timer().CaptureTimestamp(RequestTimers::Kind::RECV_START);
  }

  uint8_t* buf = reinterpret_cast<uint8_t*>(contents);
  size_t result_bytes = size * nmemb;
  std::copy(
      buf, buf + result_bytes,
      std::back_inserter(*request->infer_response_buffer_));

  // ResponseHandler may be called multiple times so we overwrite
  // RECV_END so that we always have the time of the last.
  request->Timer().CaptureTimestamp(RequestTimers::Kind::RECV_END);

  return result_bytes;
}

void
InferenceServerHttpClient::PrepareRequestJson(
    const InferOptions& options, const std::vector<InferInput*>& inputs,
    const std::vector<const InferRequestedOutput*>& outputs,
    rapidjson::Document* request_json)
{
  // Populate the request JSON.
  rapidjson::Document::AllocatorType& allocator = request_json->GetAllocator();
  request_json->SetObject();
  {
    rapidjson::Value request_id(options.request_id_.c_str(), allocator);
    request_json->AddMember("id", request_id, allocator);
    rapidjson::Value parameters(rapidjson::kObjectType);
    {
      if (options.sequence_id_ != 0) {
        rapidjson::Value sequence_id(options.sequence_id_);
        parameters.AddMember("sequence_id", sequence_id, allocator);
        rapidjson::Value sequence_start(options.sequence_start_);
        parameters.AddMember("sequence_start", sequence_start, allocator);
        rapidjson::Value sequence_end(options.sequence_end_);
        parameters.AddMember("sequence_end", sequence_end, allocator);
      }

      if (options.priority_ != 0) {
        rapidjson::Value priority(options.priority_);
        parameters.AddMember("priority", priority, allocator);
      }

      if (options.timeout_ != 0) {
        rapidjson::Value timeout(options.timeout_);
        parameters.AddMember("timeout", timeout, allocator);
      }
    }
    request_json->AddMember("parameters", parameters, allocator);
  }

  rapidjson::Value inputs_json(rapidjson::kArrayType);
  {
    for (const auto this_input : inputs) {
      rapidjson::Value this_input_json(rapidjson::kObjectType);
      {
        rapidjson::Value name(this_input->Name().c_str(), allocator);
        this_input_json.AddMember("name", name, allocator);
        rapidjson::Value shape(rapidjson::kArrayType);
        {
          for (const auto dim : this_input->Shape()) {
            rapidjson::Value dim_json(dim);
            shape.PushBack(dim_json, allocator);
          }
        }
        this_input_json.AddMember("shape", shape, allocator);
        rapidjson::Value datatype(this_input->Datatype().c_str(), allocator);
        this_input_json.AddMember("datatype", datatype, allocator);
        rapidjson::Value parameters(rapidjson::kObjectType);
        if (this_input->IsSharedMemory()) {
          std::string region_name;
          size_t offset;
          size_t byte_size;
          this_input->SharedMemoryInfo(&region_name, &byte_size, &offset);
          {
            rapidjson::Value shared_memory_region(
                region_name.c_str(), allocator);
            parameters.AddMember(
                "shared_memory_region", shared_memory_region, allocator);
            rapidjson::Value shared_memory_byte_size(byte_size);
            parameters.AddMember(
                "shared_memory_byte_size", shared_memory_byte_size, allocator);
            if (offset != 0) {
              rapidjson::Value shared_memory_offset(offset);
              parameters.AddMember(
                  "shared_memory_offset", shared_memory_offset, allocator);
            }
          }
        } else {
          size_t byte_size;
          this_input->ByteSize(&byte_size);
          rapidjson::Value binary_data_size(byte_size);
          parameters.AddMember("binary_data_size", binary_data_size, allocator);
        }
        this_input_json.AddMember("parameters", parameters, allocator);
      }
      inputs_json.PushBack(this_input_json, allocator);
    }
  }
  request_json->AddMember("inputs", inputs_json, allocator);

  rapidjson::Value ouputs_json(rapidjson::kArrayType);
  {
    for (const auto this_output : outputs) {
      rapidjson::Value this_output_json(rapidjson::kObjectType);
      {
        rapidjson::Value name(this_output->Name().c_str(), allocator);
        this_output_json.AddMember("name", name, allocator);
        rapidjson::Value parameters(rapidjson::kObjectType);
        size_t class_count = this_output->ClassCount();
        if (class_count != 0) {
          rapidjson::Value classification(class_count);
          parameters.AddMember("classification", classification, allocator);
        }
        if (this_output->IsSharedMemory()) {
          std::string region_name;
          size_t offset;
          size_t byte_size;
          this_output->SharedMemoryInfo(&region_name, &byte_size, &offset);
          {
            rapidjson::Value shared_memory_region(
                region_name.c_str(), allocator);
            parameters.AddMember(
                "shared_memory_region", shared_memory_region, allocator);
            rapidjson::Value shared_memory_byte_size(byte_size);
            parameters.AddMember(
                "shared_memory_byte_size", shared_memory_byte_size, allocator);
            if (offset != 0) {
              rapidjson::Value shared_memory_offset(offset);
              parameters.AddMember(
                  "shared_memory_offset", shared_memory_offset, allocator);
            }
          }
        } else {
          rapidjson::Value binary_data(true);
          parameters.AddMember("binary_data", binary_data, allocator);
        }
        this_output_json.AddMember("parameters", parameters, allocator);
      }
      ouputs_json.PushBack(this_output_json, allocator);
    }
  }
  request_json->AddMember("outputs", ouputs_json, allocator);
}

Error
InferenceServerHttpClient::PreRunProcessing(
    std::string& request_uri, const InferOptions& options,
    const std::vector<InferInput*>& inputs,
    const std::vector<const InferRequestedOutput*>& outputs,
    const Headers& headers, const Parameters& query_params,
    std::shared_ptr<InferRequest>& request)
{
  rapidjson::Document request_json;
  PrepareRequestJson(options, inputs, outputs, &request_json);

  // Prepare the request object to provide the data for inference.
  std::shared_ptr<HttpInferRequest> http_request =
      std::static_pointer_cast<HttpInferRequest>(request);
  http_request->InitializeRequest(request_json);

  // Add the buffers holding input tensor data
  for (const auto this_input : inputs) {
    if (!this_input->IsSharedMemory()) {
      this_input->PrepareForRequest();
      bool end_of_input = false;
      while (!end_of_input) {
        const uint8_t* buf;
        size_t buf_size;
        this_input->GetNext(&buf, &buf_size, &end_of_input);
        if (buf != nullptr) {
          http_request->AddInput(const_cast<uint8_t*>(buf), buf_size);
        }
      }
    }
  }

  // Prepare curl
  CURL* curl = http_request->easy_handle_;
  if (!curl) {
    return Error("failed to initialize HTTP client");
  }

  if (!query_params.empty()) {
    request_uri = request_uri + "?" + GetQueryString(query_params);
  }

  curl_easy_setopt(curl, CURLOPT_URL, request_uri.c_str());
  curl_easy_setopt(curl, CURLOPT_USERAGENT, "libcurl-agent/1.0");
  curl_easy_setopt(curl, CURLOPT_POST, 1L);
  curl_easy_setopt(curl, CURLOPT_TCP_NODELAY, 1L);
  if (verbose_) {
    curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
  }

  const long buffer_byte_size = 16 * 1024 * 1024;
  curl_easy_setopt(curl, CURLOPT_UPLOAD_BUFFERSIZE, buffer_byte_size);
  curl_easy_setopt(curl, CURLOPT_BUFFERSIZE, buffer_byte_size);

  // request data provided by InferRequestProvider()
  curl_easy_setopt(curl, CURLOPT_READFUNCTION, InferRequestProvider);
  curl_easy_setopt(curl, CURLOPT_READDATA, http_request.get());

  // response headers handled by InferResponseHeaderHandler()
  curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, InferResponseHeaderHandler);
  curl_easy_setopt(curl, CURLOPT_HEADERDATA, http_request.get());

  // response data handled by InferResponseHandler()
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, InferResponseHandler);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, http_request.get());

  const curl_off_t post_byte_size = http_request->total_input_byte_size_;
  curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE_LARGE, post_byte_size);

  struct curl_slist* list = nullptr;

  std::string infer_hdr{std::string(kInferHeaderContentLengthHTTPHeader) +
                        ": " +
                        std::to_string(http_request->request_json_.GetSize())};
  list = curl_slist_append(list, infer_hdr.c_str());
  list = curl_slist_append(list, "Expect:");
  list = curl_slist_append(list, "Content-Type: application/octet-stream");
  for (const auto& pr : headers) {
    std::string hdr = pr.first + ": " + pr.second;
    list = curl_slist_append(list, hdr.c_str());
  }
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, list);


  // The list will be freed when the request is destructed
  http_request->header_list_ = list;

  return Error::Success;
}

Error
InferenceServerHttpClient::Get(
    std::string& request_uri, const Headers& headers,
    const Parameters& query_params, rapidjson::Document* response,
    long* http_code)
{
  if (!query_params.empty()) {
    request_uri = request_uri + "?" + GetQueryString(query_params);
  }

  if (!curl_global.Status().IsOk()) {
    return curl_global.Status();
  }

  CURL* curl = curl_easy_init();
  if (!curl) {
    return Error("failed to initialize HTTP client");
  }

  curl_easy_setopt(curl, CURLOPT_URL, request_uri.c_str());
  curl_easy_setopt(curl, CURLOPT_USERAGENT, "libcurl-agent/1.0");
  if (verbose_) {
    curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
  }

  // Response data handled by ResponseHandler()
  std::string response_string;
  response_string.reserve(256);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, ResponseHandler);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_string);

  // Add user provided headers...
  struct curl_slist* header_list = nullptr;
  for (const auto& pr : headers) {
    std::string hdr = pr.first + ": " + pr.second;
    header_list = curl_slist_append(header_list, hdr.c_str());
  }

  if (header_list != nullptr) {
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, header_list);
  }

  CURLcode res = curl_easy_perform(curl);
  if (res != CURLE_OK) {
    curl_slist_free_all(header_list);
    curl_easy_cleanup(curl);
    return Error("HTTP client failed: " + std::string(curl_easy_strerror(res)));
  }

  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, http_code);

  curl_slist_free_all(header_list);
  curl_easy_cleanup(curl);

  if (!response_string.empty()) {
    response->Parse(response_string.c_str(), response_string.size());
    if (response->HasParseError()) {
      return Error(
          "failed to parse the request JSON buffer: " +
          std::string(GetParseError_En(response->GetParseError())) + " at " +
          std::to_string(response->GetErrorOffset()));
    }
    if (verbose_) {
      std::cout << GetJsonText(*response) << std::endl;
    }

    const auto& itr = response->FindMember("error");
    if (itr != response->MemberEnd()) {
      return Error(itr->value.GetString());
    }
  }

  return Error::Success;
}

size_t
InferenceServerHttpClient::ResponseHandler(
    void* contents, size_t size, size_t nmemb, void* userp)
{
  std::string* response_string = reinterpret_cast<std::string*>(userp);
  uint8_t* buf = reinterpret_cast<uint8_t*>(contents);
  size_t result_bytes = size * nmemb;
  std::copy(buf, buf + result_bytes, std::back_inserter(*response_string));
  return result_bytes;
}

//==============================================================================

Error
InferResultHttp::Create(
    InferResult** infer_result, std::unique_ptr<std::string> response,
    size_t json_response_size)
{
  *infer_result = reinterpret_cast<InferResult*>(
      new InferResultHttp(std::move(response), json_response_size));
  return Error::Success;
}

Error
InferResultHttp::ModelName(std::string* name) const
{
  const auto& itr = response_json_.FindMember("model_name");
  if (itr != response_json_.MemberEnd()) {
    *name = std::string(itr->value.GetString(), itr->value.GetStringLength());
  } else {
    return Error("model name was not returned in the response");
  }
  return Error::Success;
}

Error
InferResultHttp::ModelVersion(std::string* version) const
{
  const auto& itr = response_json_.FindMember("model_version");
  if (itr != response_json_.MemberEnd()) {
    *version =
        std::string(itr->value.GetString(), itr->value.GetStringLength());
  } else {
    return Error("model version was not returned in the response");
  }
  return Error::Success;
}

Error
InferResultHttp::Id(std::string* id) const
{
  const auto& itr = response_json_.FindMember("id");
  if (itr != response_json_.MemberEnd()) {
    *id = std::string(itr->value.GetString(), itr->value.GetStringLength());
  } else {
    return Error("model version was not returned in the response");
  }
  return Error::Success;
}

Error
InferResultHttp::Shape(
    const std::string& output_name, std::vector<int64_t>* shape) const
{
  shape->clear();
  auto itr = output_name_to_result_map_.find(output_name);
  if (itr != output_name_to_result_map_.end()) {
    const auto shape_itr = itr->second->FindMember("shape");
    if (shape_itr != itr->second->MemberEnd()) {
      const rapidjson::Value& shape_json = shape_itr->value;
      for (rapidjson::SizeType i = 0; i < shape_json.Size(); i++) {
        shape->push_back(shape_json[i].GetInt());
      }
    } else {
      return Error(
          "The response does not contain shape for output name " + output_name);
    }
  } else {
    return Error(
        "The response does not contain results or output name " + output_name);
  }
  return Error::Success;
}

Error
InferResultHttp::Datatype(
    const std::string& output_name, std::string* datatype) const
{
  auto itr = output_name_to_result_map_.find(output_name);
  if (itr != output_name_to_result_map_.end()) {
    const auto datatype_itr = itr->second->FindMember("datatype");
    if (datatype_itr != itr->second->MemberEnd()) {
      const rapidjson::Value& datatype_json = datatype_itr->value;
      *datatype = std::string(
          datatype_json.GetString(), datatype_json.GetStringLength());
    } else {
      return Error(
          "The response does not contain datatype for output name " +
          output_name);
    }
  } else {
    return Error(
        "The response does not contain datatype or output name " + output_name);
  }
  return Error::Success;
}


Error
InferResultHttp::RawData(
    const std::string& output_name, const uint8_t** buf,
    size_t* byte_size) const
{
  auto itr = output_name_to_buffer_map_.find(output_name);
  if (itr != output_name_to_buffer_map_.end()) {
    *buf = itr->second.first;
    *byte_size = itr->second.second;
  } else {
    return Error(
        "The response does not contain results or output name " + output_name);
  }

  return Error::Success;
}

InferResultHttp::InferResultHttp(
    std::unique_ptr<std::string> response, size_t json_response_size)
    : response_(std::move(response))
{
  size_t offset = json_response_size;
  response_json_.Parse((char*)response_.get()->c_str(), json_response_size);
  const auto& itr = response_json_.FindMember("outputs");
  if (itr != response_json_.MemberEnd()) {
    const rapidjson::Value& outputs = itr->value;
    for (size_t i = 0; i < outputs.Size(); i++) {
      const rapidjson::Value& output = outputs[i];
      const char* output_name = output["name"].GetString();
      output_name_to_result_map_[output_name] = &output;
      const auto& pitr = output.FindMember("parameters");
      if (pitr != output.MemberEnd()) {
        const rapidjson::Value& param = pitr->value;
        const auto& bitr = param.FindMember("binary_data_size");
        if (bitr != param.MemberEnd()) {
          size_t byte_size = bitr->value.GetInt();
          output_name_to_buffer_map_.emplace(
              output_name,
              std::pair<const uint8_t*, const size_t>(
                  (uint8_t*)(response_.get()->c_str()) + offset, byte_size));
          offset += byte_size;
        }
      }
    }
  }
}

std::string
InferResultHttp::DebugString() const
{
  return GetJsonText(response_json_);
}

Error
InferResultHttp::RequestStatus() const
{
  const auto& itr = response_json_.FindMember("error");
  if (itr != response_json_.MemberEnd()) {
    return Error(
        std::string(itr->value.GetString(), itr->value.GetStringLength()));
  }

  return Error::Success;
}

//==============================================================================

}}}  // namespace nvidia::inferenceserver::client
