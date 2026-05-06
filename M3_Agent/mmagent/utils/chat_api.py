# Copyright (2025) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import openai
from concurrent.futures import ThreadPoolExecutor
from time import sleep
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Disable httpx logging
logging.getLogger("httpx").setLevel(logging.CRITICAL)
# Disable urllib3 logging (which httpx uses)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
# Disable httpcore logging (which httpx uses)
logging.getLogger("httpcore").setLevel(logging.CRITICAL)

# api utils

processing_config = json.load(open("configs/processing_config.json"))
temp = processing_config["temperature"]

config = json.load(open("configs/api_config.json"))
client = {}
client_init_errors = {}

for model_name, model_cfg in config.items():
    try:
        api_key = model_cfg.get("api_key")
        azure_endpoint = model_cfg.get("azure_endpoint")
        api_version = model_cfg.get("api_version")
        base_url = model_cfg.get("base_url") or model_cfg.get("openai_base_url")

        if azure_endpoint:
            if not api_version:
                raise ValueError("Missing 'api_version' for Azure client")
            client[model_name] = openai.AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_version=api_version,
                api_key=api_key,
            )
        else:
            # OpenAI-compatible endpoint (or official OpenAI when base_url is empty)
            if base_url:
                client[model_name] = openai.OpenAI(api_key=api_key, base_url=base_url)
            else:
                client[model_name] = openai.OpenAI(api_key=api_key)
    except Exception as e:
        client_init_errors[model_name] = f"{type(e).__name__}: {e}"
        logger.error(f"Failed to initialize client for model '{model_name}': {e}")

def _get_client_or_raise(model):
    if model in client:
        return client[model]
    available = sorted(client.keys())
    init_err = client_init_errors.get(model)
    msg = (
        f"Model client '{model}' is not initialized. "
        f"Available model clients: {available}."
    )
    if init_err:
        msg += f" Init error for this model: {init_err}"
    raise KeyError(msg)

MAX_RETRIES = 5

def get_response(model, messages, timeout=30):
    """Get chat completion response from specified model.

    Args:
        model (str): Model identifier
        messages (list): List of message dictionaries

    Returns:
        tuple: (response content, total tokens used)
    """
    model_client = _get_client_or_raise(model)
    response = model_client.chat.completions.create(
        model=model, messages=messages, temperature=temp, timeout=timeout, max_tokens=8192
    )
    
    # return answer and number of tokens
    return response.choices[0].message.content, response.usage.total_tokens

def get_response_with_retry(model, messages, timeout=30):
    """Retry get_response up to MAX_RETRIES times with error handling.

    Args:
        model (str): Model identifier
        messages (list): List of message dictionaries

    Returns:
        tuple: (response content, total tokens used)
        
    Raises:
        Exception: If all retries fail
    """
    for i in range(MAX_RETRIES):
        try:
            return get_response(model, messages, timeout)
        except Exception as e:
            sleep(20)
            logger.warning(f"Retry {i} times, exception: {e} from message {messages}")
            continue
    raise Exception(f"Failed to get response after {MAX_RETRIES} retries")

def parallel_get_response(model, messages, timeout=30):
    """Process multiple messages in parallel using ThreadPoolExecutor.
    Messages are processed in batches, with each batch completing before starting the next.

    Args:
        model (str): Model identifier
        messages (list): List of message lists to process

    Returns:
        tuple: (list of responses, total tokens used)
    """
    batch_size = config[model]["qpm"]
    responses = []
    total_tokens = 0

    for i in range(0, len(messages), batch_size):
        batch = messages[i:i + batch_size]
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            batch_responses = list(executor.map(lambda msg: get_response_with_retry(model, msg, timeout), batch))
            
        # Extract answers and tokens from batch responses
        batch_answers = [response[0] for response in batch_responses]
        batch_tokens = [response[1] for response in batch_responses]
        
        responses.extend(batch_answers)
        total_tokens += sum(batch_tokens)

    return responses, total_tokens


def get_embedding(model, text, timeout=15):
    """Get embedding for text using specified model.

    Args:
        model (str): Model identifier
        text (str): Text to embed

    Returns:
        tuple: (embedding vector, total tokens used)
    """
    model_client = _get_client_or_raise(model)
    response = model_client.embeddings.create(input=text, model=model, timeout=timeout)
    return response.data[0].embedding, response.usage.total_tokens


def get_embedding_with_retry(model, text, timeout=15):
    """Retry get_embedding up to MAX_RETRIES times with error handling.

    Args:
        model (str): Model identifier
        text (str): Text to embed

    Returns:
        tuple: (embedding vector, total tokens used)
        
    Raises:
        Exception: If all retries fail
    """
    last_exception = None
    for i in range(MAX_RETRIES):
        try:
            return get_embedding(model, text, timeout)
        except Exception as e:
            last_exception = e
            sleep(20)
            logger.warning(f"Retry {i} times, exception: {e} from get embedding")
            continue
    if last_exception is not None:
        raise Exception(
            f"Failed to get embedding after {MAX_RETRIES} retries for model '{model}'. "
            f"Last error: {type(last_exception).__name__}: {last_exception}"
        ) from last_exception
    raise Exception(f"Failed to get embedding after {MAX_RETRIES} retries for model '{model}'")

def parallel_get_embedding(model, texts, timeout=15):
    """Process multiple texts in parallel to get embeddings.

    Args:
        model (str): Model identifier
        texts (list): List of texts to embed

    Returns:
        tuple: (list of embeddings, total tokens used)
    """
    batch_size = config[model]["qpm"]
    embeddings = []
    total_tokens = 0
    
    # Process texts in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        max_workers = len(batch)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(lambda x: get_embedding_with_retry(model, x, timeout), batch))
            
        # Split batch results into embeddings and tokens
        batch_embeddings = [result[0] for result in results]
        batch_tokens = [result[1] for result in results]
        
        embeddings.extend(batch_embeddings)
        total_tokens += sum(batch_tokens)
        
    return embeddings, total_tokens

def get_whisper(model, file_path):
    """Transcribe audio file using Whisper model.

    Args:
        model (str): Model identifier
        file_path (str): Path to audio file

    Returns:
        str: Transcription text
    """
    file = open(file_path, "rb")
    model_client = _get_client_or_raise(model)
    return model_client.audio.transcriptions.create(model=model, file=file).text

def get_whisper_with_retry(model, file_path):
    """Retry Whisper transcription with error handling.

    Args:
        model (str): Model identifier
        file_path (str): Path to audio file

    Returns:
        str: Transcription text
        
    Raises:
        Exception: If all retries fail
    """
    for i in range(MAX_RETRIES):
        try:
            return get_whisper(model, file_path)
        except Exception as e:
            sleep(20)
            logger.warning(f"Retry {i} times, exception: {e}")
    raise Exception(f"Failed to get response after {MAX_RETRIES} retries")

def parallel_get_whisper(model, file_paths):
    """Process multiple audio files in parallel using Whisper model.

    Args:
        model (str): Model identifier
        file_paths (list): List of audio file paths

    Returns:
        list: List of transcription results
    """
    batch_size = config[model]["qpm"]
    responses = []
    
    for i in range(0, len(file_paths), batch_size):
        batch = file_paths[i:i + batch_size]
        max_workers = len(batch)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            batch_responses = list(executor.map(lambda x: get_whisper_with_retry(model, x), batch))
            
        responses.extend(batch_responses)
        
    return responses

def generate_messages(inputs):
    """Generate message list for chat completion from mixed inputs.

    Args:
        inputs (list): List of input dictionaries with 'type' and 'content' keys
        type can be:
            "text" - text content
            "image/jpeg", "image/png" - base64 encoded images
            "video/mp4", "video/webm" - base64 encoded videos
            "video_url" - video URL
            "audio/mp3", "audio/wav" - base64 encoded audio
        content should be a string for text,
        a list of base64 encoded media for images/video/audio,
        or a string (url) for video_url
        inputs are like: 
        [
            {
                "type": "video_base64/mp4",
                "content": <base64>
            },
            {
                "type": "text",
                "content": "Describe the video content."
            },
            ...
        ]

    Returns:
        list: Formatted messages for chat completion
    """
    messages = []
    messages.append(
        {"role": "system", "content": "You are an expert in video understanding."}
    )
    content = []
    for input in inputs:
        if not input["content"]:
            logger.warning("empty content, skip")
            continue
        if input["type"] == "text":
            content.append({"type": "text", "text": input["content"]})
        elif input["type"] in ["images/jpeg", "images/png"]:
            img_format = input["type"].split("/")[1]
            if isinstance(input["content"][0], str):
                content.extend(
                    [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{img_format};base64,{img}",
                                "detail": "high",
                            },
                        }
                        for img in input["content"]
                    ]
                )
            else:
                for img in input["content"]:
                    content.append({
                        "type": "text",
                        "text": img[0],
                    })
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{img_format};base64,{img[1]}",
                            "detail": "high",
                        },
                    })
        elif input["type"] == "video_url":
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": input["content"]},
                }
            )
        elif input["type"] in ["video_base64/mp4", "video_base64/webm"]:
            video_format = input["type"].split("/")[1]
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:video/{video_format};base64,{input['content']}"},
                }
            )
        elif input["type"] in ["audio_base64/mp3", "audio_base64/wav"]:
            audio_format = input["type"].split("/")[1]
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:audio/{audio_format};base64,{input['content']}"
                    },
                }
            )
        else:
            raise ValueError(f"Invalid input type: {input['type']}")
    messages.append({"role": "user", "content": content})
    return messages

def print_messages(messages):
    for message in messages:
        if message["role"] == "user":
            for item in message["content"]:
                if item["type"] == "text":
                    logger.debug(item['text'])
