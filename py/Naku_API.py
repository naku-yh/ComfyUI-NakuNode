"""
Naku API Nodes - 集成各种 API 服务的节点
"""

import torch
import numpy as np
from PIL import Image
import base64
import requests
import json
import os
import io
import time
from io import BytesIO
import comfy.utils
import uuid
import folder_paths

import concurrent.futures  # Add this import for the multi-image processing
import re  # Add this import for pattern matching
import numpy as np
from PIL import Image


def pil2tensor_naku(image):
    """
    Convert PIL image(s) to tensor, matching ComfyUI's implementation.

    Args:
        image: Single PIL Image or list of PIL Images

    Returns:
        torch.Tensor: Image tensor with values normalized to [0, 1]
    """
    import torch
    if isinstance(image, list):
        if len(image) == 0:
            return torch.empty(0)
        return torch.cat([pil2tensor_naku(img) for img in image], dim=0)

    if image.mode == 'RGBA':
        image = image.convert('RGB')
    elif image.mode != 'RGB':
        image = image.convert('RGB')

    img_array = np.array(image).astype(np.float32) / 255.0

    return torch.from_numpy(img_array)[None,]


def tensor2pil_naku(image):
    """
    Convert tensor to PIL image(s), matching ComfyUI's implementation.

    Args:
        image: Tensor with shape [B, H, W, 3] or [H, W, 3], values in range [0, 1]

    Returns:
        List[Image.Image]: List of PIL Images
    """
    import torch
    batch_count = image.size(0) if len(image.shape) > 3 else 1
    if batch_count > 1:
        out = []
        for i in range(batch_count):
            out.extend(tensor2pil_naku(image[i]))
        return out

    numpy_image = np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)

    return [Image.fromarray(numpy_image)]


# Define API category
CATEGORY_TYPE = "NakuNodes/API"

# Global variable for baseurl that will be shared
global_naku_baseurl = "https://ai.comfly.chat"

# Configuration functions
def get_baseurl():
    """Return the API base URL"""
    global global_naku_baseurl
    return global_naku_baseurl

def get_config():
    """获取配置 - This is a simplified version that returns empty config"""
    try:
        import os
        config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Comflyapi.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        else:
            # Try looking in parent directory
            parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            config_path = os.path.join(parent_dir, 'Comflyapi.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                return config
            else:
                return {}
    except:
        return {}

def save_config(config):
    """保存配置 - placeholder"""
    pass

# Mock ComflyVideoAdapter class
class ComflyVideoAdapter:
    def __init__(self, video_url):
        self.video_url = video_url
        # Set default dimensions for video objects
        self.width = 1920
        self.height = 1080

        # Initialize attributes that may be accessed by ComfyUI video nodes
        self.images = None  # Will be populated when needed
        self.audio = None   # Will be populated when needed
        self.frame_rate = 30.0  # Default frame rate

    def get_dimensions(self):
        """Return video dimensions as (width, height) tuple"""
        return (self.width, self.height)

    def get_components(self):
        """Return self to provide images, audio and frame_rate attributes directly"""
        # Return self so that the ComflyVideoAdapter object itself provides the needed attributes
        return self

    def save_to(self, path, format=None, **kwargs):
        """Download and save the video from the URL to the specified path"""
        import requests
        import os
        from urllib.parse import urlparse
        import mimetypes

        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Check if video_url is valid
        if not self.video_url or not self.video_url.strip():
            print(f"Warning: Attempting to save video with empty URL")
            # Create an empty file to avoid breaking the workflow
            with open(path, 'w') as f:
                f.write("")
            return path

        try:
            # Download the video from the URL
            response = requests.get(self.video_url, stream=True)
            response.raise_for_status()

            # Determine file extension from URL or content-type header
            if '?' in self.video_url:
                # Remove query parameters
                clean_url = self.video_url.split('?')[0]
            else:
                clean_url = self.video_url

            # Get extension from URL
            parsed_url = urlparse(clean_url)
            ext = os.path.splitext(parsed_url.path)[1]

            if not ext:
                # Try to determine from content-type header
                content_type = response.headers.get('content-type', '')
                ext = mimetypes.guess_extension(content_type.split(';')[0]) or '.mp4'

            # Use format parameter if provided, otherwise use detected extension
            if format:
                ext = f".{format.lower()}"

            # Update path with correct extension if needed
            if not path.endswith(ext):
                base_path = os.path.splitext(path)[0]
                path = f"{base_path}{ext}"

            # Write the video content to the file
            with open(path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            return path

        except Exception as e:
            print(f"Error downloading video from {self.video_url}: {str(e)}")
            # If download fails, create an empty file to avoid breaking the workflow
            with open(path, 'w') as f:
                f.write("")
            return path

    def __str__(self):
        return f"ComflyVideoAdapter(video_url={self.video_url})"

    def __repr__(self):
        return self.__str__()

# --- Google Veo3 Node ---
class NakuNodeAPI_Googel_Veo3:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (["veo3", "veo3-fast", "veo3-pro", "veo3-fast-frames", "veo3-pro-frames", "veo3.1", "veo3.1-pro", "veo3.1-components"], {"default": "veo3"}),
                "enhance_prompt": ("BOOLEAN", {"default": False}),
                "aspect_ratio": (["16:9", "9:16"], {"default": "16:9"}),
            },
            "optional": {
                "apikey": ("STRING", {"default": ""}),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "enable_upsample": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "response")
    FUNCTION = "generate_video"
    CATEGORY = CATEGORY_TYPE

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def image_to_base64(self, image_tensor):
        """Convert tensor to base64 string"""
        if image_tensor is None:
            return None

        pil_image = tensor2pil_naku(image_tensor)[0]

        # Resize image if too large to prevent "request entity too large" error
        # Maintain aspect ratio by resizing the longest side to 1920 pixels
        max_dimension = 1920
        original_width, original_height = pil_image.size

        if original_width > max_dimension or original_height > max_dimension:
            # Calculate the scaling factor to maintain aspect ratio
            scale_factor = max_dimension / max(original_width, original_height)
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)

            # Resize the image using LANCZOS resampling for better quality
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def generate_video(self, prompt, model="veo3", enhance_prompt=False, aspect_ratio="16:9", apikey="",
                      image1=None, image2=None, image3=None, seed=0, enable_upsample=False):

        if apikey.strip():
            self.api_key = apikey

        if not self.api_key:
            error_response = {"code": "error", "message": "API key not found in Comflyapi.json"}
            return ("", "", json.dumps(error_response))

        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            has_images = any(img is not None for img in [image1, image2, image3])

            payload = {
                "prompt": prompt,
                "model": model,
                "enhance_prompt": enhance_prompt
            }

            if seed > 0:
                payload["seed"] = seed

            if model in ["veo3", "veo3-fast", "veo3-pro", "veo3.1", "veo3.1-pro", "veo3.1-components"] and aspect_ratio:
                payload["aspect_ratio"] = aspect_ratio

            if model in ["veo3", "veo3-fast", "veo3-pro", "veo3.1", "veo3.1-pro", "veo3.1-components"] and enable_upsample:
                payload["enable_upsample"] = enable_upsample

            if has_images:
                images_base64 = []
                for img in [image1, image2, image3]:
                    if img is not None:
                        batch_size = img.shape[0]
                        for i in range(batch_size):
                            single_image = img[i:i+1]
                            image_base64 = self.image_to_base64(single_image)
                            if image_base64:
                                images_base64.append(f"data:image/png;base64,{image_base64}")

                if images_base64:
                    payload["images"] = images_base64

            response = requests.post(
                f"{get_baseurl()}/google/v1/models/veo/videos",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )

            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(error_message)
                return ("", "", json.dumps({"code": "error", "message": error_message}))

            result = response.json()

            if result.get("code") != "success":
                error_message = f"API returned error: {result.get('message', 'Unknown error')}"
                print(error_message)
                return ("", "", json.dumps({"code": "error", "message": error_message}))

            task_id = result.get("data")
            if not task_id:
                error_message = "No task ID returned from API"
                print(error_message)
                return ("", "", json.dumps({"code": "error", "message": error_message}))

            pbar.update_absolute(30)

            max_attempts = 150
            attempts = 0
            video_url = None

            while attempts < max_attempts:
                time.sleep(2)
                attempts += 1

                try:
                    status_response = requests.get(
                        f"{get_baseurl()}/google/v1/tasks/{task_id}",
                        headers=self.get_headers(),
                        timeout=self.timeout
                    )

                    if status_response.status_code != 200:
                        continue

                    status_result = status_response.json()

                    if status_result.get("code") != "success":
                        continue

                    data = status_result.get("data", {})
                    status = data.get("status", "")
                    progress = data.get("progress", "0%")

                    try:
                        if progress.endswith('%'):
                            progress_num = int(progress.rstrip('%'))
                            pbar_value = min(90, 30 + progress_num * 60 / 100)
                            pbar.update_absolute(pbar_value)
                    except (ValueError, AttributeError):
                        progress_value = min(80, 30 + (attempts * 50 // max_attempts))
                        pbar.update_absolute(progress_value)

                    if status == "SUCCESS":
                        if "data" in data and "video_url" in data["data"]:
                            video_url = data["data"]["video_url"]
                            break
                    elif status == "FAILURE":
                        fail_reason = data.get("fail_reason", "Unknown error")
                        error_message = f"Video generation failed: {fail_reason}"
                        print(error_message)
                        return ("", "", json.dumps({"code": "error", "message": error_message}))

                except Exception as e:
                    print(f"Error checking generation status: {str(e)}")

            if not video_url:
                error_message = "Failed to retrieve video URL after multiple attempts"
                print(error_message)
                return ("", "", json.dumps({"code": "error", "message": error_message}))

            if video_url:
                pbar.update_absolute(95)

                response_data = {
                    "code": "success",
                    "task_id": task_id,
                    "prompt": prompt,
                    "model": model,
                    "enhance_prompt": enhance_prompt,
                    "aspect_ratio": aspect_ratio if model in ["veo3", "veo3-fast", "veo3-pro"] else "default",
                    "enable_upsample": enable_upsample if model in ["veo3", "veo3-fast", "veo3-pro"] else False,
                    "video_url": video_url,
                    "images_count": len([img for img in [image1, image2, image3] if img is not None])
                }
                video_adapter = ComflyVideoAdapter(video_url)
                return (video_adapter, video_url, json.dumps(response_data))

        except Exception as e:
            error_message = f"Error generating video: {str(e)}"
            print(error_message)
            return ("", "", json.dumps({"code": "error", "message": error_message}))


# --- Google Nano Banana Nodes ---
class NakuNodeAPI_nano_banana:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "model": (["nano-banana-2","gemini-3-pro-image-preview", "gemini-2.5-flash-image", "nano-banana", "nano-banana-hd", "gemini-2.5-flash-image-preview"], {"default": "nano-banana"}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05}),
                "apikey": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "max_tokens": ("INT", {"default": 32768, "min": 1, "max": 32768})
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "response", "image_url")
    FUNCTION = "process"
    CATEGORY = CATEGORY_TYPE

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def image_to_base64(self, image_tensor):
        """Convert tensor to base64 string with data URI prefix"""
        if image_tensor is None:
            return None

        pil_image = tensor2pil_naku(image_tensor)[0]

        # Resize image if too large to prevent "request entity too large" error
        # Maintain aspect ratio by resizing the longest side to 1920 pixels
        max_dimension = 1920
        original_width, original_height = pil_image.size

        if original_width > max_dimension or original_height > max_dimension:
            # Calculate the scaling factor to maintain aspect ratio
            scale_factor = max_dimension / max(original_width, original_height)
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)

            # Resize the image using LANCZOS resampling for better quality
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return base64_str

    def send_request_streaming(self, payload):
        """Send a streaming request to the API"""
        full_response = ""
        session = requests.Session()

        try:
            response = session.post(
                f"{get_baseurl()}/v1/chat/completions",
                headers=self.get_headers(),
                json=payload,
                stream=True,
                timeout=self.timeout
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    line_text = line.decode('utf-8').strip()
                    if line_text.startswith('data: '):
                        data = line_text[6:]
                        if data == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data)
                            if 'choices' in chunk and chunk['choices']:
                                delta = chunk['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    content = delta['content']
                                    full_response += content
                        except json.JSONDecodeError:
                            continue

            return full_response

        except requests.exceptions.Timeout:
            raise TimeoutError(f"API request timed out after {self.timeout} seconds")
        except Exception as e:
            raise Exception(f"Error in streaming response: {str(e)}")

    def process(self, text, model="gemini-2.5-flash-image-preview",
                image1=None, image2=None, image3=None, image4=None,
                temperature=1.0, top_p=0.95, apikey="", seed=0, max_tokens=32768):

        if apikey.strip():
            self.api_key = apikey

        default_image = None
        for img in [image1, image2, image3, image4]:
            if img is not None:
                default_image = img
                break

        if default_image is None:
            blank_image = Image.new('RGB', (512, 512), color='white')
            default_image = pil2tensor_naku(blank_image)

        try:
            if not self.api_key:
                return (default_image, "API key not provided. Please set your API key.", "")

            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            content = [{"type": "text", "text": text}]

            images_added = 0
            for idx, img in enumerate([image1, image2, image3, image4], 1):
                if img is not None:
                    batch_size = img.shape[0]
                    print(f"Processing image{idx} with {batch_size} batch size")

                    for i in range(batch_size):
                        single_image = img[i:i+1]
                        image_base64 = self.image_to_base64(single_image)
                        if image_base64:
                            content.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                            })
                            images_added += 1

            print(f"Total of {images_added} images added to the request")

            messages = [{
                "role": "user",
                "content": content
            }]

            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "stream": True
            }

            if seed > 0:
                payload["seed"] = seed

            pbar.update_absolute(30)

            try:
                response_text = self.send_request_streaming(payload)
                pbar.update_absolute(70)
            except Exception as e:
                error_message = f"API Error: {str(e)}"
                print(error_message)
                return (default_image, error_message, "")

            base64_pattern = r'data:image\/[^;]+;base64,([A-Za-z0-9+/=]+)'
            base64_matches = re.findall(base64_pattern, response_text)

            if base64_matches:
                try:
                    image_data = base64.b64decode(base64_matches[0])
                    generated_image = Image.open(BytesIO(image_data))
                    generated_tensor = pil2tensor_naku(generated_image)

                    pbar.update_absolute(100)
                    return (generated_tensor, response_text, f"data:image/png;base64,{base64_matches[0]}")
                except Exception as e:
                    print(f"Error processing base64 image data: {str(e)}")

            image_pattern = r'!\[.*?\]\((.*?)\)'
            matches = re.findall(image_pattern, response_text)

            if not matches:
                url_pattern = r'https?://\S+\.(?:jpg|jpeg|png|gif|webp)'
                matches = re.findall(url_pattern, response_text)

            if not matches:
                all_urls_pattern = r'https?://\S+'
                matches = re.findall(all_urls_pattern, response_text)

            if matches:
                image_url = matches[0]
                try:
                    img_response = requests.get(image_url, timeout=self.timeout)
                    img_response.raise_for_status()

                    generated_image = Image.open(BytesIO(img_response.content))
                    generated_tensor = pil2tensor_naku(generated_image)

                    pbar.update_absolute(100)
                    return (generated_tensor, response_text, image_url)
                except Exception as e:
                    print(f"Error downloading image: {str(e)}")
                    return (default_image, f"{response_text}\n\nError downloading image: {str(e)}", image_url)
            else:
                pbar.update_absolute(100)
                return (default_image, response_text, "")

        except Exception as e:
            error_message = f"Error processing request: {str(e)}"
            print(error_message)
            return (default_image, error_message, "")


class NakuNodeAPI_nano_banana_edit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "mode": (["text2img", "img2img"], {"default": "text2img"}),
                "model": (["nano-banana", "nano-banana-hd"], {"default": "nano-banana"}),
                "aspect_ratio": (["16:9", "4:3", "4:5", "3:2", "1:1", "2:3", "3:4", "5:4", "9:16", "21:9"], {"default": "1:1"}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "apikey": ("STRING", {"default": ""}),
                "response_format": (["url", "b64_json"], {"default": "url"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "webhook": ("STRING", {"default": ""})
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "response")
    FUNCTION = "generate_image"
    CATEGORY = CATEGORY_TYPE

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 600

    def get_headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}"
        }

    def image_to_base64(self, image_tensor):
        """Convert tensor to base64 string"""
        if image_tensor is None:
            return None

        pil_image = tensor2pil_naku(image_tensor)[0]

        # Resize image if too large to prevent "request entity too large" error
        # Maintain aspect ratio by resizing the longest side to 1920 pixels
        max_dimension = 1920
        original_width, original_height = pil_image.size

        if original_width > max_dimension or original_height > max_dimension:
            # Calculate the scaling factor to maintain aspect ratio
            scale_factor = max_dimension / max(original_width, original_height)
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)

            # Resize the image using LANCZOS resampling for better quality
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def _download_image_content(self, url):
        """Helper to download image bytes"""
        try:
            resp = requests.get(url, timeout=self.timeout)
            resp.raise_for_status()
            return resp.content, url, None
        except Exception as e:
            return None, url, str(e)

    def generate_image(self, prompt, mode="text2img", model="nano-banana", aspect_ratio="1:1",
                      image1=None, image2=None, image3=None, image4=None,
                      apikey="", response_format="url", seed=0, webhook=""):
        if apikey.strip():
            self.api_key = apikey

        if not self.api_key:
            error_message = "API key not found in Comflyapi.json"
            print(error_message)
            blank_image = Image.new('RGB', (1024, 1024), color='white')
            blank_tensor = pil2tensor_naku(blank_image)
            return (blank_tensor, error_message)

        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            final_prompt = prompt

            query_params = {"async": "true"}
            if webhook:
                query_params["webhook"] = webhook

            if mode == "text2img":
                headers = self.get_headers()
                headers["Content-Type"] = "application/json"

                payload = {
                    "prompt": final_prompt,
                    "model": model,
                    "aspect_ratio": aspect_ratio
                }

                if response_format:
                    payload["response_format"] = response_format

                if seed > 0:
                    payload["seed"] = seed

                response = requests.post(
                    f"{get_baseurl()}/v1/images/generations",
                    headers=headers,
                    json=payload,
                    params=query_params,
                    timeout=self.timeout
                )
            else:
                headers = self.get_headers()

                files = []
                for img in [image1, image2, image3, image4]:
                    if img is not None:
                        pil_img = tensor2pil_naku(img)[0]
                        buffered = BytesIO()
                        pil_img.save(buffered, format="PNG")
                        buffered.seek(0)
                        files.append(('image', ('image.png', buffered, 'image/png')))

                data = {
                    "prompt": final_prompt,
                    "model": model,
                    "aspect_ratio": aspect_ratio
                }

                if response_format:
                    data["response_format"] = response_format

                if seed > 0:
                    data["seed"] = str(seed)

                response = requests.post(
                    f"{get_baseurl()}/v1/images/edits",
                    headers=headers,
                    data=data,
                    files=files,
                    params=query_params,
                    timeout=self.timeout
                )

            pbar.update_absolute(30)

            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(error_message)
                blank_image = Image.new('RGB', (1024, 1024), color='white')
                blank_tensor = pil2tensor_naku(blank_image)
                return (blank_tensor, error_message)

            result = response.json()

            if "task_id" not in result:
                error_message = "No task_id in response"
                print(error_message)
                blank_image = Image.new('RGB', (1024, 1024), color='white')
                blank_tensor = pil2tensor_naku(blank_image)
                return (blank_tensor, error_message)

            task_id = result["task_id"]
            print(f"[NakuNodeAPI_nano_banana_edit] Task submitted successfully. Task ID: {task_id}")

            pbar.update_absolute(40)

            max_retries = 150
            retry_count = 0

            while retry_count < max_retries:
                time.sleep(2)
                retry_count += 1

                try:
                    status_response = requests.get(
                        f"{get_baseurl()}/v1/images/tasks/{task_id}",
                        headers=self.get_headers(),
                        timeout=self.timeout
                    )

                    if status_response.status_code != 200:
                        print(f"[NakuNodeAPI_nano_banana_edit] Status check failed with code: {status_response.status_code}")
                        continue

                    status_result = status_response.json()

                    outer_data = status_result.get("data", {})
                    status = outer_data.get("status", "")
                    progress = outer_data.get("progress", "0%")

                    pbar.update_absolute(40 + min(50, retry_count * 50 // max_retries))

                    if status == "SUCCESS":
                        inner_data = outer_data.get("data", {})
                        image_list = inner_data.get("data", [])

                        if not image_list:
                            error_message = "No image data in successful response"
                            print(f"[NakuNodeAPI_nano_banana_edit] {error_message}")
                            blank_image = Image.new('RGB', (1024, 1024), color='white')
                            blank_tensor = pil2tensor_naku(blank_image)
                            return (blank_tensor, error_message)

                        generated_tensors = []
                        response_info = f"Generated {len(image_list)} images using {model}\n"
                        response_info += f"Aspect ratio: {aspect_ratio}\n"
                        response_info += f"Task ID: {task_id}\n"

                        if seed > 0:
                            response_info += f"Seed: {seed}\n"

                        print(f"\n[NakuNodeAPI_nano_banana_edit] ========== Generated Image URLs ==========")

                        urls_to_download = []
                        b64_items = []

                        for i, item in enumerate(image_list):
                            if "b64_json" in item and item["b64_json"]:
                                b64_items.append((i, item["b64_json"]))
                            elif "url" in item and item["url"]:
                                urls_to_download.append((i, item["url"]))

                        results_map = {}

                        for idx, b64_data in b64_items:
                            try:
                                image_data_bytes = base64.b64decode(b64_data)
                                generated_image = Image.open(BytesIO(image_data_bytes))
                                generated_tensor = pil2tensor_naku(generated_image)
                                results_map[idx] = generated_tensor
                                response_info += f"Image {idx+1}: Base64 data\n"
                                print(f"[NakuNodeAPI_nano_banana_edit] Image {idx+1}: Base64 encoded data processed")
                            except Exception as e:
                                print(f"Error processing base64 image {idx+1}: {e}")

                        if urls_to_download:
                            print(f"[NakuNodeAPI_nano_banana_edit] Downloading {len(urls_to_download)} images in parallel...")
                            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                                future_to_idx = {executor.submit(self._download_image_content, url): (idx, url) for idx, url in urls_to_download}

                                for future in concurrent.futures.as_completed(future_to_idx):
                                    idx, url = future_to_idx[future]
                                    try:
                                        content, _, err = future.result()
                                        if err:
                                            print(f"[NakuNodeAPI_nano_banana_edit] Error downloading image {idx+1}: {err}")
                                            continue

                                        generated_image = Image.open(BytesIO(content))
                                        generated_tensor = pil2tensor_naku(generated_image)
                                        results_map[idx] = generated_tensor
                                        response_info += f"Image {idx+1}: {url}\n"
                                        print(f"[NakuNodeAPI_nano_banana_edit] Image {idx+1} downloaded successfully")
                                    except Exception as exc:
                                        print(f"[NakuNodeAPI_nano_banana_edit] Exception processing image {idx+1}: {exc}")

                        for i in range(len(image_list)):
                            if i in results_map:
                                generated_tensors.append(results_map[i])

                        print(f"[NakuNodeAPI_nano_banana_edit] ==========================================\n")

                        pbar.update_absolute(100)

                        if generated_tensors:
                            combined_tensor = torch.cat(generated_tensors, dim=0)
                            return (combined_tensor, response_info)
                        else:
                            error_message = "Failed to process any images"
                            print(error_message)
                            blank_image = Image.new('RGB', (1024, 1024), color='white')
                            blank_tensor = pil2tensor_naku(blank_image)
                            return (blank_tensor, error_message)

                    elif status == "FAILURE":
                        fail_reason = outer_data.get("fail_reason", "Unknown error")
                        error_message = f"Task failed: {fail_reason}"
                        print(f"[NakuNodeAPI_nano_banana_edit] {error_message}")
                        blank_image = Image.new('RGB', (1024, 1024), color='white')
                        blank_tensor = pil2tensor_naku(blank_image)
                        return (blank_tensor, error_message)

                except Exception as e:
                    print(f"[NakuNodeAPI_nano_banana_edit] Error checking task status: {str(e)}")
                    continue

            error_message = f"Task timed out after {max_retries} retries"
            print(f"[NakuNodeAPI_nano_banana_edit] {error_message}")
            blank_image = Image.new('RGB', (1024, 1024), color='white')
            blank_tensor = pil2tensor_naku(blank_image)
            return (blank_tensor, error_message)

        except Exception as e:
            error_message = f"Error in image generation: {str(e)}"
            print(f"[NakuNodeAPI_nano_banana_edit] {error_message}")
            blank_image = Image.new('RGB', (1024, 1024), color='white')
            blank_tensor = pil2tensor_naku(blank_image)
            return (blank_tensor, error_message)


class NakuNodeAPI_nano_banana2_edit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "mode": (["text2img", "img2img"], {"default": "text2img"}),
                "model": (["nano-banana-2", "nano-banana-2-2k", "nano-banana-2-4k"], {"default": "nano-banana-2"}),
                "aspect_ratio": (["auto", "16:9", "4:3", "4:5", "3:2", "1:1", "2:3", "3:4", "5:4", "9:16", "21:9"], {"default": "auto"}),
                "image_size": (["1K", "2K", "4K"], {"default": "2K"}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "image7": ("IMAGE",),
                "image8": ("IMAGE",),
                "image9": ("IMAGE",),
                "image10": ("IMAGE",),
                "image11": ("IMAGE",),
                "image12": ("IMAGE",),
                "image13": ("IMAGE",),
                "image14": ("IMAGE",),
                "apikey": ("STRING", {"default": ""}),
                "response_format": (["url", "b64_json"], {"default": "url"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "webhook": ("STRING", {"default": ""})
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "response", "image_url")
    FUNCTION = "generate_image"
    CATEGORY = CATEGORY_TYPE

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 600

    def get_headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}"
        }

    def image_to_base64(self, image_tensor):
        """Convert tensor to base64 string"""
        if image_tensor is None:
            return None

        pil_image = tensor2pil_naku(image_tensor)[0]

        # Resize image if too large to prevent "request entity too large" error
        # Maintain aspect ratio by resizing the longest side to 1920 pixels
        max_dimension = 1920
        original_width, original_height = pil_image.size

        if original_width > max_dimension or original_height > max_dimension:
            # Calculate the scaling factor to maintain aspect ratio
            scale_factor = max_dimension / max(original_width, original_height)
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)

            # Resize the image using LANCZOS resampling for better quality
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def _download_image_content(self, url):
        """Helper to download image bytes"""
        try:
            resp = requests.get(url, timeout=self.timeout)
            resp.raise_for_status()
            return resp.content, url, None
        except Exception as e:
            return None, url, str(e)

    def generate_image(self, prompt, mode="text2img", model="nano-banana-2", aspect_ratio="auto",
                      image_size="2K", image1=None, image2=None, image3=None, image4=None,
                      image5=None, image6=None, image7=None, image8=None, image9=None,
                      image10=None, image11=None, image12=None, image13=None, image14=None,
                      apikey="", response_format="url", seed=0, webhook=""):

        if apikey.strip():
            self.api_key = apikey

        if not self.api_key:
            error_message = "API key not found in Comflyapi.json"
            print(error_message)
            blank_image = Image.new('RGB', (1024, 1024), color='white')
            blank_tensor = pil2tensor_naku(blank_image)
            return (blank_tensor, error_message, "")

        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            final_prompt = prompt

            query_params = {"async": "true"}
            if webhook:
                query_params["webhook"] = webhook

            image_count = 0

            if mode == "text2img":
                headers = self.get_headers()
                headers["Content-Type"] = "application/json"

                payload = {
                    "prompt": final_prompt,
                    "model": model,
                    "aspect_ratio": aspect_ratio
                }

                if model == "nano-banana-2":
                    payload["image_size"] = image_size

                if response_format:
                    payload["response_format"] = response_format

                if seed > 0:
                    payload["seed"] = seed

                response = requests.post(
                    f"{get_baseurl()}/v1/images/generations",
                    headers=headers,
                    json=payload,
                    params=query_params,
                    timeout=self.timeout
                )
            else:
                headers = self.get_headers()

                all_images = [image1, image2, image3, image4, image5, image6, image7,
                             image8, image9, image10, image11, image12, image13, image14]

                files = []
                for img in all_images:
                    if img is not None:
                        pil_img = tensor2pil_naku(img)[0]
                        buffered = BytesIO()
                        pil_img.save(buffered, format="PNG")
                        buffered.seek(0)
                        files.append(('image', (f'image_{image_count}.png', buffered, 'image/png')))
                        image_count += 1

                print(f"[NakuNodeAPI_nano_banana2_edit] Processing {image_count} input images")

                data = {
                    "prompt": final_prompt,
                    "model": model,
                    "aspect_ratio": aspect_ratio
                }

                if model == "nano-banana-2":
                    data["image_size"] = image_size

                if response_format:
                    data["response_format"] = response_format

                if seed > 0:
                    data["seed"] = str(seed)

                response = requests.post(
                    f"{get_baseurl()}/v1/images/edits",
                    headers=headers,
                    data=data,
                    files=files,
                    params=query_params,
                    timeout=self.timeout
                )

            pbar.update_absolute(30)

            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(error_message)
                blank_image = Image.new('RGB', (1024, 1024), color='white')
                blank_tensor = pil2tensor_naku(blank_image)
                return (blank_tensor, error_message, "")

            result = response.json()

            if "task_id" not in result:
                error_message = "No task_id in response"
                print(error_message)
                blank_image = Image.new('RGB', (1024, 1024), color='white')
                blank_tensor = pil2tensor_naku(blank_image)
                return (blank_tensor, error_message, "")

            task_id = result["task_id"]
            print(f"[NakuNodeAPI_nano_banana2_edit] Task submitted successfully. Task ID: {task_id}")

            pbar.update_absolute(40)

            max_retries = 150
            retry_count = 0

            while retry_count < max_retries:
                time.sleep(2)
                retry_count += 1

                try:
                    status_response = requests.get(
                        f"{get_baseurl()}/v1/images/tasks/{task_id}",
                        headers=self.get_headers(),
                        timeout=self.timeout
                    )

                    if status_response.status_code != 200:
                        print(f"[NakuNodeAPI_nano_banana2_edit] Status check failed with code: {status_response.status_code}")
                        continue

                    status_result = status_response.json()

                    outer_data = status_result.get("data", {})
                    status = outer_data.get("status", "")
                    progress = outer_data.get("progress", "0%")

                    pbar.update_absolute(40 + min(50, retry_count * 50 // max_retries))

                    if status == "SUCCESS":
                        inner_data = outer_data.get("data", {})
                        image_list = inner_data.get("data", [])

                        if not image_list:
                            error_message = "No image data in successful response"
                            print(f"[NakuNodeAPI_nano_banana2_edit] {error_message}")
                            blank_image = Image.new('RGB', (1024, 1024), color='white')
                            blank_tensor = pil2tensor_naku(blank_image)
                            return (blank_tensor, error_message, "")

                        generated_tensors = []
                        first_image_url = ""

                        response_info = f"Generated {len(image_list)} images using {model}\n"

                        if model == "nano-banana-2":
                            response_info += f"Image size: {image_size}\n"

                        response_info += f"Aspect ratio: {aspect_ratio}\n"
                        response_info += f"Task ID: {task_id}\n"

                        if mode == "img2img":
                            response_info += f"Input images: {image_count}\n"

                        if seed > 0:
                            response_info += f"Seed: {seed}\n"

                        print(f"\n[NakuNodeAPI_nano_banana2_edit] ========== Generated Image URLs ==========")

                        urls_to_download = []
                        b64_items = []
                        results_map = {}

                        for i, item in enumerate(image_list):
                            if "b64_json" in item and item["b64_json"]:
                                b64_items.append((i, item["b64_json"]))
                            elif "url" in item and item["url"]:
                                urls_to_download.append((i, item["url"]))
                                if i == 0: first_image_url = item["url"]

                        for idx, b64_data in b64_items:
                            try:
                                image_data_bytes = base64.b64decode(b64_data)
                                generated_image = Image.open(BytesIO(image_data_bytes))
                                generated_tensor = pil2tensor_naku(generated_image)
                                results_map[idx] = generated_tensor
                                response_info += f"Image {idx+1}: Base64 data\n"
                                print(f"[NakuNodeAPI_nano_banana2_edit] Image {idx+1}: Base64 encoded data processed")
                            except Exception as e:
                                print(f"Error processing base64 image {idx+1}: {e}")

                        if urls_to_download:
                            print(f"[NakuNodeAPI_nano_banana2_edit] Downloading {len(urls_to_download)} images in parallel...")
                            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                                future_to_idx = {executor.submit(self._download_image_content, url): (idx, url) for idx, url in urls_to_download}

                                for future in concurrent.futures.as_completed(future_to_idx):
                                    idx, url = future_to_idx[future]
                                    try:
                                        content, _, err = future.result()
                                        if err:
                                            print(f"[NakuNodeAPI_nano_banana2_edit] Error downloading image {idx+1}: {err}")
                                            continue

                                        generated_image = Image.open(BytesIO(content))
                                        generated_tensor = pil2tensor_naku(generated_image)
                                        results_map[idx] = generated_tensor
                                        response_info += f"Image {idx+1}: {url}\n"
                                        print(f"[NakuNodeAPI_nano_banana2_edit] Image {idx+1} downloaded successfully")
                                    except Exception as exc:
                                        print(f"[NakuNodeAPI_nano_banana2_edit] Exception processing image {idx+1}: {exc}")

                        for i in range(len(image_list)):
                            if i in results_map:
                                generated_tensors.append(results_map[i])

                        print(f"[NakuNodeAPI_nano_banana2_edit] ==========================================\n")

                        pbar.update_absolute(100)

                        if generated_tensors:
                            combined_tensor = torch.cat(generated_tensors, dim=0)
                            return (combined_tensor, response_info, first_image_url)
                        else:
                            error_message = "Failed to process any images"
                            print(error_message)
                            blank_image = Image.new('RGB', (1024, 1024), color='white')
                            blank_tensor = pil2tensor_naku(blank_image)
                            return (blank_tensor, error_message, "")

                    elif status == "FAILURE":
                        fail_reason = outer_data.get("fail_reason", "Unknown error")
                        error_message = f"Task failed: {fail_reason}"
                        print(f"[NakuNodeAPI_nano_banana2_edit] {error_message}")
                        blank_image = Image.new('RGB', (1024, 1024), color='white')
                        blank_tensor = pil2tensor_naku(blank_image)
                        return (blank_tensor, error_message, "")

                except Exception as e:
                    print(f"[NakuNodeAPI_nano_banana2_edit] Error checking task status: {str(e)}")
                    continue

            error_message = f"Task timed out after {max_retries} retries"
            print(f"[NakuNodeAPI_nano_banana2_edit] {error_message}")
            blank_image = Image.new('RGB', (1024, 1024), color='white')
            blank_tensor = pil2tensor_naku(blank_image)
            return (blank_tensor, error_message, "")

        except Exception as e:
            error_message = f"Error in image generation: {str(e)}"
            print(f"[NakuNodeAPI_nano_banana2_edit] {error_message}")
            blank_image = Image.new('RGB', (1024, 1024), color='white')
            blank_tensor = pil2tensor_naku(blank_image)
            return (blank_tensor, error_message, "")


class NakuNodeAPI_GeminiTextOnly:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": ("STRING", {"default": "gemini-2.5-pro"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "video": ("VIDEO",),
                "api_key": ("STRING", {"default": ""}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 8192}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "generate_text"
    CATEGORY = CATEGORY_TYPE

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 120

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def tensor_to_base64(self, tensor):
        if tensor is None:
            return None
        if tensor.dtype != torch.uint8:
            tensor = (tensor * 255).clamp(0, 255).byte()
        tensor = tensor.cpu()
        if tensor.shape[-1] == 3:
            img = Image.fromarray(tensor.numpy(), 'RGB')
        else:
            img = Image.fromarray(tensor.numpy(), 'RGBA')
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def generate_text(self, prompt, model, temperature, top_p, max_tokens, seed, image=None, video=None, api_key=""):
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key
            save_config(config)

        if not self.api_key:
            return ("API key not found in Comflyapi.json",)

        try:
            content = [{"type": "text", "text": prompt}]

            if video is not None:
                video_url = getattr(video, 'video_url', None)
                if video_url:
                    content.append({
                        "type": "video_url",
                        "video_url": {"url": video_url}
                    })
            elif image is not None:
                if len(image.shape) == 4:
                    image = image[0]
                img_b64 = self.tensor_to_base64(image)
                if img_b64:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                    })

            messages = [{"role": "user", "content": content}]

            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "seed": seed if seed > 0 else None
            }

            response = requests.post(
                f"{get_baseurl()}/v1/chat/completions",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            text = result["choices"][0]["message"]["content"]
            return (text,)

        except Exception as e:
            return (f"Error: {str(e)}",)


# --- Kling Nodes ---
class NakuNodeAPI_kling_text2video:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model_name": (["kling-v2-1-master", "kling-v2-master", "kling-v1-6", "kling-v1-5", "kling-v1"], {"default": "kling-v1-6"}),
                "imagination": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "aspect_ratio": (["1:1", "16:9", "9:16"], {"default": "1:1"}),
                "mode": (["std", "pro"], {"default": "std"}),
                "duration": (["5", "10"], {"default": "5"}),
                "num_videos": ("INT", {"default": 1, "min": 1, "max": 4}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647})
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "camera": (["none", "horizontal", "vertical", "zoom", "vertical_shake", "horizontal_shake",
                          "rotate", "master_down_zoom", "master_zoom_up", "master_right_rotate_zoom",
                          "master_left_rotate_zoom"], {"default": "none"}),
                "camera_value": ("FLOAT", {"default": 0, "min": -10, "max": 10, "step": 0.1})
            }
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "task_id", "video_id", "response")
    FUNCTION = "generate_video"
    CATEGORY = CATEGORY_TYPE

    def __init__(self):
        super().__init__()
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300
        self.model_capabilities = {
            "kling-v1": {
                "std": {
                    "5": {"video": True, "camera": True},
                    "10": {"video": True, "camera": False}
                },
                "pro": {
                    "5": {"video": True, "camera": False},
                    "10": {"video": True, "camera": False}
                }
            },
            "kling-v1-5": {
                "std": {
                    "5": {"video": False, "camera": False},
                    "10": {"video": False, "camera": False}
                },
                "pro": {
                    "5": {"video": False, "camera": True},
                    "10": {"video": False, "camera": False}
                }
            },
            "kling-v1-6": {
                "std": {
                    "5": {"video": True, "camera": False},
                    "10": {"video": True, "camera": False}
                },
                "pro": {
                    "5": {"video": False, "camera": False},
                    "10": {"video": False, "camera": False}
                }
            },
            "kling-v2-master": {
                "std": {
                    "5": {"video": False, "camera": False},
                    "10": {"video": False, "camera": False}
                },
                "pro": {
                    "5": {"video": False, "camera": False},
                    "10": {"video": False, "camera": False}
                }
            }
        }

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def get_camera_json(self, camera, camera_value=0):
        camera_mappings = {
            "none": {"type":"empty","horizontal":0,"vertical":0,"zoom":0,"tilt":0,"pan":0,"roll":0},
            "horizontal": {"type":"horizontal","horizontal":camera_value,"vertical":0,"zoom":0,"tilt":0,"pan":0,"roll":0},
            "vertical": {"type":"vertical","horizontal":0,"vertical":camera_value,"zoom":0,"tilt":0,"pan":0,"roll":0},
            "zoom": {"type":"zoom","horizontal":0,"vertical":0,"zoom":camera_value,"tilt":0,"pan":0,"roll":0},
            "vertical_shake": {"type":"vertical_shake","horizontal":0,"vertical":camera_value,"zoom":0.5,"tilt":0,"pan":0,"roll":0},
            "horizontal_shake": {"type":"horizontal_shake","horizontal":camera_value,"vertical":0,"zoom":0.5,"tilt":0,"pan":0,"roll":0},
            "rotate": {"type":"rotate","horizontal":0,"vertical":0,"zoom":0,"tilt":0,"pan":camera_value,"roll":0},
            "master_down_zoom": {"type":"zoom","horizontal":0,"vertical":0,"zoom":camera_value,"tilt":camera_value,"pan":0,"roll":0},
            "master_zoom_up": {"type":"zoom","horizontal":0.2,"vertical":0,"zoom":camera_value,"tilt":0,"pan":0,"roll":0},
            "master_right_rotate_zoom": {"type":"rotate","horizontal":0,"vertical":0,"zoom":camera_value,"tilt":0,"pan":0,"roll":camera_value},
            "master_left_rotate_zoom": {"type":"rotate","horizontal":0,"vertical":0,"zoom":camera_value,"tilt":0,"pan":camera_value,"roll":0},
        }
        return json.dumps(camera_mappings.get(camera, camera_mappings["none"]))

    def generate_video(self, prompt, model_name, imagination, aspect_ratio, mode, duration, num_videos,
                  negative_prompt="", camera="none", camera_value=0, seed=0, api_key=""):
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key

        if not self.api_key:
            error_response = {"task_status": "failed", "task_status_msg": "API key not found in Comflyapi.json"}
            return ("", "", "", "", json.dumps(error_response))

        camera_json = {}
        if model_name == "kling-v1":
            camera_json = self.get_camera_json(camera, camera_value)
        else:
            camera_json = self.get_camera_json("none", 0)

        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": "",
            "image_tail": "",
            "aspect_ratio": aspect_ratio,
            "duration": duration,
            "model_name": model_name,
            "imagination": imagination,
            "num_videos": num_videos,
            "camera_json": camera_json,
            "seed": seed
        }

        if model_name != "kling-v2-master":
            payload["mode"] = mode
        else:
            print("Note: kling-v2-master model doesn't use mode parameter")

        try:
            pbar = comfy.utils.ProgressBar(100)
            response = requests.post(
                f"{get_baseurl()}/kling/v1/videos/text2video",
                headers=self.get_headers(),
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            if result["code"] != 0:
                error_response = {"task_status": "failed", "task_status_msg": f"API Error: {result['message']}"}
                return ("", "", "", "", json.dumps(error_response))

            task_id = result["data"]["task_id"]
            pbar.update_absolute(5)

            last_status = {}
            while True:
                time.sleep(2)
                status_response = requests.get(
                    f"{get_baseurl()}/kling/v1/videos/text2video/{task_id}",
                    headers=self.get_headers()
                )
                status_response.raise_for_status()
                status_result = status_response.json()
                last_status = status_result["data"]

                progress = status_result["data"].get("progress", 0)
                pbar.update_absolute(progress)

                if status_result["data"]["task_status"] == "succeed":
                    pbar.update_absolute(100)
                    video_url = status_result["data"]["task_result"]["videos"][0]["url"]
                    video_id = status_result["data"]["task_result"]["videos"][0]["id"]

                    response_data = {
                        "task_status": "succeed",
                        "task_status_msg": "Video generated successfully",
                        "progress": 100,
                        "video_url": video_url
                    }

                    video_adapter = ComflyVideoAdapter(video_url)
                    return (video_adapter, video_url, task_id, video_id, json.dumps(response_data))

                elif status_result["data"]["task_status"] == "failed":
                    error_msg = status_result["data"].get("task_status_msg", "Unknown error")
                    error_response = {
                        "task_status": "failed",
                        "task_status_msg": error_msg,
                    }
                    print(f"Task failed: {error_msg}")
                    return ("", "", task_id, "", json.dumps(error_response))
        except Exception as e:
            error_response = {"task_status": "failed", "task_status_msg": f"Error generating video: {str(e)}"}
            print(f"Error generating video: {str(e)}")
            return ("", "", "", "", json.dumps(error_response))


class NakuNodeAPI_kling_image2video:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
                "model_name": (["kling-v2-1", "kling-v2-1-master", "kling-v2-master", "kling-v1-6", "kling-v1-5", "kling-v1"], {"default": "kling-v1-6"}),
                "imagination": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "aspect_ratio": (["1:1", "16:9", "9:16"], {"default": "1:1"}),
                "mode": (["std", "pro"], {"default": "std"}),
                "duration": (["5", "10"], {"default": "5"}),
                "num_videos": ("INT", {"default": 1, "min": 1, "max": 4}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647})
            },
            "optional": {
                "image_tail": ("IMAGE",),
                "api_key": ("STRING", {"default": ""}),
                "camera": (["none", "horizontal", "vertical", "zoom", "vertical_shake", "horizontal_shake",
                          "rotate", "master_down_zoom", "master_zoom_up", "master_right_rotate_zoom",
                          "master_left_rotate_zoom"], {"default": "none"}),
                "camera_value": ("FLOAT", {"default": 0, "min": -10, "max": 10, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "task_id", "video_id", "response")
    FUNCTION = "generate_video"
    CATEGORY = CATEGORY_TYPE

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300
        self.model_compatibility = {
            "kling-v1": {
                "std": {"5": True, "10": True},
                "pro": {"5": True, "10": True}
            },
            "kling-v1-5": {
                "std": {"5": False, "10": False},
                "pro": {"5": True, "10": True}
            },
            "kling-v1-6": {
                "std": {"5": False, "10": False},
                "pro": {"5": True, "10": True}
            },
            "kling-v2-master": {
                "std": {"5": False, "10": False},
                "pro": {"5": False, "10": False}
            },
            "kling-v2-1": {
                "std": {"5": False, "10": False},
                "pro": {"5": True, "10": True}
            }
        }

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def check_tail_image_compatibility(self, model_name, mode, duration):
        try:
            if model_name == "kling-v2-1":
                return mode == "pro"

            return self.model_compatibility.get(model_name, {}).get(mode, {}).get(duration, False)
        except:
            return False

    def generate_video(self, image, prompt, model_name, imagination, aspect_ratio, mode, duration,
                  num_videos, negative_prompt="", camera="none", camera_value=0, seed=0, image_tail=None, api_key=""):
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key

        if not self.api_key:
            error_response = {"task_status": "failed", "task_status_msg": "API key not found in Comflyapi.json"}
            return ("", "", "", "", json.dumps(error_response))

        has_tail_image = image_tail is not None

        if has_tail_image:
            check_mode = "std" if model_name == "kling-v2-master" else mode
            tail_compatible = self.check_tail_image_compatibility(model_name, check_mode, duration)
            if not tail_compatible:
                warning_message = f"Warning: model/mode/duration({model_name}/{mode if model_name != 'kling-v2-master' else 'N/A'}/{duration}) does not support using both image and image_tail."
                print(warning_message)

                if model_name == "kling-v1-5" or model_name == "kling-v1-6":
                    if mode == "std":
                        suggestion = "\nSuggestion: Try switching to 'pro' mode which supports tail images."
                        warning_message += suggestion

                error_response = {
                    "task_status": "failed",
                    "task_status_msg": warning_message
                }
                return ("", "", "", "", json.dumps(error_response))

        camera_json = {}
        if model_name in ["kling-v1-5", "kling-v1-6"] and mode == "pro" and camera != "none":
            camera_json = self.get_camera_json(camera, camera_value)
        else:
            camera_json = self.get_camera_json("none", 0)

        try:
            pil_image = tensor2pil_naku(image)[0]
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            buffered = BytesIO()
            pil_image.save(buffered, format="JPEG", quality=95)
            image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            image_tail_base64 = ""
            if has_tail_image:
                pil_tail = tensor2pil_naku(image_tail)[0]
                if pil_tail.mode != 'RGB':
                    pil_tail = pil_tail.convert('RGB')
                tail_buffered = BytesIO()
                pil_tail.save(tail_buffered, format="JPEG", quality=95)
                image_tail_base64 = base64.b64encode(tail_buffered.getvalue()).decode('utf-8')

            payload = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "image": image_base64,
                "image_tail": image_tail_base64,
                "aspect_ratio": aspect_ratio,
                "duration": duration,
                "model_name": model_name,
                "imagination": imagination,
                "num_videos": num_videos,
                "camera_json": camera_json if isinstance(camera_json, str) else json.dumps(camera_json),
                "seed": seed
            }

            if model_name != "kling-v2-master":
                payload["mode"] = mode

            print(f"Sending request with parameters: model={model_name}, duration={duration}, aspect_ratio={aspect_ratio}")
            if model_name != "kling-v2-master":
                print(f"Mode: {mode}")
            else:
                print("Note: kling-v2-master model doesn't use mode parameter")

            print(f"Image base64 length: {len(image_base64)}")
            if has_tail_image:
                print(f"Image tail base64 length: {len(image_tail_base64)}")

            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(5)
            response = requests.post(
                f"{get_baseurl()}/kling/v1/videos/image2video",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )

            print(f"Response status: {response.status_code}")
            if response.status_code != 200:
                error_message = f"Error: {response.status_code} {response.reason} - {response.text}"
                print(error_message)
                error_response = {"task_status": "failed", "task_status_msg": error_message}
                return ("", "", "", "", json.dumps(error_response))

            result = response.json()
            if result["code"] != 0:
                error_response = {"task_status": "failed", "task_status_msg": f"API Error: {result['message']}"}
                return ("", "", "", "", json.dumps(error_response))

            task_id = result["data"]["task_id"]
            pbar.update_absolute(10)

            last_status = {}
            while True:
                time.sleep(2)
                status_response = requests.get(
                    f"{get_baseurl()}/kling/v1/videos/image2video/{task_id}",
                    headers=self.get_headers(),
                    timeout=self.timeout
                )
                status_response.raise_for_status()
                status_result = status_response.json()
                last_status = status_result["data"]

                progress = 0
                if status_result["data"]["task_status"] == "processing":
                    progress = status_result["data"].get("progress", 50)
                elif status_result["data"]["task_status"] == "succeed":
                    progress = 100

                pbar.update_absolute(progress)

                if status_result["data"]["task_status"] == "succeed":
                    pbar.update_absolute(100)
                    video_url = status_result["data"]["task_result"]["videos"][0]["url"]
                    video_id = status_result["data"]["task_result"]["videos"][0]["id"]

                    response_data = {
                        "task_status": "succeed",
                        "task_status_msg": "Video generated successfully",
                        "progress": 100,
                        "video_url": video_url
                    }

                    video_adapter = ComflyVideoAdapter(video_url)
                    return (video_adapter, video_url, task_id, video_id, json.dumps(response_data))

                elif status_result["data"]["task_status"] == "failed":
                    error_msg = status_result["data"].get("task_status_msg", "Unknown error")
                    error_response = {
                        "task_status": "failed",
                        "task_status_msg": error_msg,
                    }
                    print(f"Task failed: {error_msg}")
                    return ("", "", task_id, "", json.dumps(error_response))
        except Exception as e:
            error_response = {"task_status": "failed", "task_status_msg": f"Error generating video: {str(e)}"}
            print(f"Error generating video: {str(e)}")
            return ("", "", "", "", json.dumps(error_response))

    def get_camera_json(self, camera, camera_value=0):
        camera_mappings = {
            "none": {"type":"empty","horizontal":0,"vertical":0,"zoom":0,"tilt":0,"pan":0,"roll":0},
            "horizontal": {"type":"horizontal","horizontal":camera_value,"vertical":0,"zoom":0,"tilt":0,"pan":0,"roll":0},
            "vertical": {"type":"vertical","horizontal":0,"vertical":camera_value,"zoom":0,"tilt":0,"pan":0,"roll":0},
            "zoom": {"type":"zoom","horizontal":0,"vertical":0,"zoom":camera_value,"tilt":0,"pan":0,"roll":0},
            "vertical_shake": {"type":"vertical_shake","horizontal":0,"vertical":camera_value,"zoom":0.5,"tilt":0,"pan":0,"roll":0},
            "horizontal_shake": {"type":"horizontal_shake","horizontal":camera_value,"vertical":0,"zoom":0.5,"tilt":0,"pan":0,"roll":0},
            "rotate": {"type":"rotate","horizontal":0,"vertical":0,"zoom":0,"tilt":0,"pan":camera_value,"roll":0},
            "master_down_zoom": {"type":"zoom","horizontal":0,"vertical":0,"zoom":camera_value,"tilt":camera_value,"pan":0,"roll":0},
            "master_zoom_up": {"type":"zoom","horizontal":0.2,"vertical":0,"zoom":camera_value,"tilt":0,"pan":0,"roll":0},
            "master_right_rotate_zoom": {"type":"rotate","horizontal":0,"vertical":0,"zoom":camera_value,"tilt":0,"pan":0,"roll":camera_value},
            "master_left_rotate_zoom": {"type":"rotate","horizontal":0,"vertical":0,"zoom":camera_value,"tilt":0,"pan":camera_value,"roll":0},
        }
        return json.dumps(camera_mappings.get(camera, camera_mappings["none"]))


class NakuNodeAPI_kling_multi_image2video:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model_name": (["kling-v1-6"], {"default": "kling-v1-6"}),
                "mode": (["std", "pro"], {"default": "std"}),
                "duration": (["5", "10"], {"default": "5"}),
                "aspect_ratio": (["16:9", "9:16", "1:1"], {"default": "16:9"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "api_key": ("STRING", {"default": ""}),
                "max_retries": ("INT", {"default": 10, "min": 1, "max": 30}),
                "initial_timeout": ("INT", {"default": 600, "min": 30, "max": 900}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647})
            }
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "task_id", "video_id", "response")
    FUNCTION = "generate_video"
    CATEGORY = CATEGORY_TYPE

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300
        self.session = requests.Session()
        retry_strategy = requests.packages.urllib3.util.retry.Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def image_to_base64(self, image_tensor):
        """Convert tensor to base64 string without data URI prefix"""
        if image_tensor is None:
            return None

        try:
            pil_image = tensor2pil_naku(image_tensor)[0]

            # Resize image if too large to prevent "request entity too large" error
            # Maintain aspect ratio by resizing the longest side to 1920 pixels
            max_dimension = 1920
            original_width, original_height = pil_image.size

            if original_width > max_dimension or original_height > max_dimension:
                # Calculate the scaling factor to maintain aspect ratio
                scale_factor = max_dimension / max(original_width, original_height)
                new_width = int(original_width * scale_factor)
                new_height = int(original_height * scale_factor)

                # Resize the image using LANCZOS resampling for better quality
                pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            buffered = BytesIO()
            pil_image.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"Error converting image to base64: {str(e)}")
            return None

    def make_request_with_retry(self, method, url, **kwargs):
        max_retries = kwargs.pop('max_retries', 10)
        initial_timeout = kwargs.pop('initial_timeout', self.timeout)

        for attempt in range(1, max_retries + 1):
            current_timeout = min(initial_timeout * (2 ** (attempt - 1)), 900)

            try:
                kwargs['timeout'] = current_timeout

                if method.lower() == 'get':
                    response = self.session.get(url, **kwargs)
                else:
                    response = self.session.post(url, **kwargs)

                response.raise_for_status()
                return response

            except requests.exceptions.Timeout as e:
                if attempt == max_retries:
                    raise
                wait_time = min(2 ** (attempt - 1), 60)
                time.sleep(wait_time)

            except requests.exceptions.ConnectionError as e:
                if attempt == max_retries:
                    raise
                wait_time = min(2 ** (attempt - 1), 60)
                time.sleep(wait_time)

            except requests.exceptions.HTTPError as e:
                if e.response.status_code in (400, 401, 403):
                    print(f"Client error: {str(e)}")
                    raise
                if attempt == max_retries:
                    raise
                wait_time = min(2 ** (attempt - 1), 60)
                time.sleep(wait_time)

            except Exception as e:
                if attempt == max_retries:
                    raise
                wait_time = min(2 ** (attempt - 1), 60)
                time.sleep(wait_time)

    def poll_task_status(self, task_id, max_attempts=100, initial_interval=2, max_interval=60, headers=None, pbar=None):
        attempt = 0
        interval = initial_interval
        last_status = None

        while attempt < max_attempts:
            attempt += 1

            try:
                status_response = self.make_request_with_retry(
                    'get',
                    f"{get_baseurl()}/kling/v1/videos/multi-image2video/{task_id}",
                    headers=headers
                )

                status_result = status_response.json()

                if status_result["code"] != 0:
                    print(f"API returned error code: {status_result['code']} - {status_result['message']}")
                    if status_result["code"] in (400, 401, 403):
                        return {"task_status": "failed", "task_status_msg": status_result["message"]}
                    time.sleep(interval)
                    interval = min(interval * 1.5, max_interval)
                    continue

                last_status = status_result["data"]

                if pbar:
                    progress = 0
                    if last_status["task_status"] == "processing":
                        progress = 50
                    elif last_status["task_status"] == "succeed":
                        progress = 100
                    pbar.update_absolute(progress)

                if last_status["task_status"] == "succeed":
                    return last_status
                elif last_status["task_status"] == "failed":
                    return last_status

                if last_status["task_status"] == "processing":
                    interval = min(interval * 1.2, max_interval)
                else:
                    interval = max(interval * 0.8, initial_interval)

                time.sleep(interval)

            except Exception as e:
                time.sleep(interval)
                interval = min(interval * 2, max_interval)

        if last_status:
            return last_status
        else:
            return {"task_status": "failed", "task_status_msg": "Maximum polling attempts reached without getting a valid status"}

    def generate_video(self, prompt, model_name, mode, duration, aspect_ratio, negative_prompt="",
                  image1=None, image2=None, image3=None, image4=None, api_key="",
                  max_retries=10, initial_timeout=300, seed=0):
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key

        if not self.api_key:
            error_response = {"task_status": "failed", "task_status_msg": "API key not found in Comflyapi.json"}
            return ("", "", "", "", json.dumps(error_response))

        pbar = comfy.utils.ProgressBar(100)
        pbar.update_absolute(10)

        try:
            image_list = []
            for idx, img in enumerate([image1, image2, image3, image4], 1):
                if img is not None:
                    base64_str = self.image_to_base64(img)
                    if base64_str:
                        image_list.append({"image": base64_str})

            if not image_list:
                error_response = {"task_status": "failed", "task_status_msg": "No valid images provided"}
                return ("", "", "", "", json.dumps(error_response))

            payload = {
                "model_name": model_name,
                "image_list": image_list,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "mode": mode,
                "duration": duration,
                "aspect_ratio": aspect_ratio
            }

            if seed > 0:
                payload["seed"] = seed

            headers = self.get_headers()
            pbar.update_absolute(30)

            print("Submitting multi-image video generation request...")
            response = self.make_request_with_retry(
                'post',
                f"{get_baseurl()}/kling/v1/videos/multi-image2video",
                headers=headers,
                json=payload,
                max_retries=max_retries,
                initial_timeout=initial_timeout
            )

            result = response.json()

            if result["code"] != 0:
                error_message = f"API Error: {result['message']}"
                print(error_message)
                error_response = {"task_status": "failed", "task_status_msg": error_message}
                return ("", "", "", "", json.dumps(error_response))

            task_id = result["data"]["task_id"]
            pbar.update_absolute(40)
            print(f"Multi-image video generation task submitted. Task ID: {task_id}")

            print("Waiting for video generation to complete...")
            last_status = self.poll_task_status(
                task_id,
                max_attempts=100,
                initial_interval=2,
                max_interval=60,
                headers=headers,
                pbar=pbar
            )

            if last_status["task_status"] == "succeed":
                pbar.update_absolute(100)
                video_url = last_status["task_result"]["videos"][0]["url"]
                video_id = last_status["task_result"]["videos"][0]["id"]

                response_data = {
                    "task_status": "succeed",
                    "task_status_msg": "Video generated successfully",
                    "progress": 100,
                    "video_url": video_url
                }

                video_adapter = ComflyVideoAdapter(video_url)
                return (video_adapter, video_url, task_id, video_id, json.dumps(response_data))

            elif last_status["task_status"] == "failed":
                error_msg = last_status.get("task_status_msg", "Unknown error")
                error_response = {
                    "task_status": "failed",
                    "task_status_msg": error_msg,
                }
                print(f"Task failed: {error_msg}")
                return ("", "", task_id, "", json.dumps(error_response))

            else:
                error_msg = f"Unexpected task status: {last_status.get('task_status', 'unknown')}"
                error_response = {
                    "task_status": "failed",
                    "task_status_msg": error_msg,
                }
                print(f"Task error: {error_msg}")
                return ("", "", task_id, "", json.dumps(error_response))

        except Exception as e:
            import traceback
            traceback.print_exc()
            error_response = {"task_status": "failed", "task_status_msg": f"Error generating video: {str(e)}"}
            print(f"Error generating video: {str(e)}")
            return ("", "", "", "", json.dumps(error_response))


class NakuNodeAPI_video_extend:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_id": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""})
            }
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_id", "response")
    FUNCTION = "extend_video"
    CATEGORY = CATEGORY_TYPE

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300

    def extend_video(self, video_id, prompt="", api_key=""):
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key

        if not self.api_key:
            error_response = {"task_status": "failed", "task_status_msg": "API key not found in Comflyapi.json"}
            return ("", "", json.dumps(error_response))

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "video_id": video_id,
            "prompt": prompt
        }
        try:
            response = requests.post(
                f"{get_baseurl()}/kling/v1/videos/video-extend",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            if result["code"] != 0:
                error_response = {"task_status": "failed", "task_status_msg": f"API Error: {result['message']}"}
                return ("", "", json.dumps(error_response))

            task_id = result["data"]["task_id"]
            pbar = comfy.utils.ProgressBar(100)

            last_status = {}
            while True:
                time.sleep(2)
                status_response = requests.get(
                    f"{get_baseurl()}/kling/v1/videos/video-extend/{task_id}",
                    headers=headers,
                    timeout=self.timeout
                )
                status_response.raise_for_status()
                status_result = status_response.json()
                last_status = status_result["data"]

                progress = 0
                if status_result["data"]["task_status"] == "processing":
                    progress = 50
                elif status_result["data"]["task_status"] == "succeed":
                    progress = 100
                pbar.update_absolute(progress)

                if status_result["data"]["task_status"] == "succeed":
                    video_url = status_result["data"]["task_result"]["videos"][0]["url"]
                    new_video_id = status_result["data"]["task_result"]["videos"][0]["id"]

                    response_data = {
                        "task_status": "succeed",
                        "task_status_msg": "Video extended successfully",
                        "progress": 100,
                        "video_url": video_url
                    }

                    return (video_url, new_video_id, json.dumps(response_data))

                elif status_result["data"]["task_status"] == "failed":
                    error_msg = status_result["data"].get("task_status_msg", "Unknown error")
                    error_response = {
                        "task_status": "failed",
                        "task_status_msg": error_msg,
                    }
                    print(f"Task failed: {error_msg}")
                    return ("", "", json.dumps(error_response))

        except Exception as e:
            error_response = {"task_status": "failed", "task_status_msg": f"Error extending video: {str(e)}"}
            print(f"Error extending video: {str(e)}")
            return ("", "", json.dumps(error_response))


class NakuNodeAPI_lip_sync:
    @classmethod
    def INPUT_TYPES(cls):
        cls.zh_voices = [
            ["阳光少年", "genshin_vindi2"],
            ["懂事小弟", "zhinen_xuesheng"],
            ["运动少年", "tiyuxi_xuedi"],
            ["青春少女", "ai_shatang"],
            ["温柔小妹", "genshin_klee2"],
            ["元气少女", "genshin_kirara"],
            ["阳光男生", "ai_kaiya"],
            ["幽默小哥", "tiexin_nanyou"],
            ["文艺小哥", "ai_chenjiahao_712"],
            ["甜美邻家", "girlfriend_1_speech02"],
            ["温柔姐姐", "chat1_female_new-3"],
            ["职场女青", "girlfriend_2_speech02"],
            ["活泼男童", "cartoon-boy-07"],
            ["俏皮女童", "cartoon-girl-01"],
            ["稳重老爸", "ai_huangyaoshi_712"],
            ["温柔妈妈", "you_pingjing"],
            ["严肃上司", "ai_laoguowang_712"],
            ["优雅贵妇", "chengshu_jiejie"],
            ["慈祥爷爷", "zhuxi_speech02"],
            ["唠叨爷爷", "uk_oldman3"],
            ["唠叨奶奶", "laopopo_speech02"],
            ["和蔼奶奶", "heainainai_speech02"],
            ["东北老铁", "dongbeilaotie_speech02"],
            ["重庆小伙", "chongqingxiaohuo_speech02"],
            ["四川妹子", "chuanmeizi_speech02"],
            ["潮汕大叔", "chaoshandashu_speech02"],
            ["台湾男生", "ai_taiwan_man2_speech02"],
            ["西安掌柜", "xianzhanggui_speech02"],
            ["天津姐姐", "tianjinjiejie_speech02"],
            ["新闻播报男", "diyinnansang_DB_CN_M_04-v2"],
            ["译制片男", "yizhipiannan-v1"],
            ["元气少女", "guanxiaofang-v2"],
            ["撒娇女友", "tianmeixuemei-v1"],
            ["刀片烟嗓", "daopianyansang-v1"],
            ["乖巧正太", "mengwa-v1"]
        ]

        cls.en_voices = [
            ["Sunny", "genshin_vindi2"],
            ["Sage", "zhinen_xuesheng"],
            ["Ace", "AOT"],
            ["Blossom", "ai_shatang"],
            ["Peppy", "genshin_klee2"],
            ["Dove", "genshin_kirara"],
            ["Shine", "ai_kaiya"],
            ["Anchor", "oversea_male1"],
            ["Lyric", "ai_chenjiahao_712"],
            ["Melody", "girlfriend_4_speech02"],
            ["Tender", "chat1_female_new-3"],
            ["Siren", "chat_0407_5-1"],
            ["Zippy", "cartoon-boy-07"],
            ["Bud", "uk_boy1"],
            ["Sprite", "cartoon-girl-01"],
            ["Candy", "PeppaPig_platform"],
            ["Beacon", "ai_huangzhong_712"],
            ["Rock", "ai_huangyaoshi_712"],
            ["Titan", "ai_laoguowang_712"],
            ["Grace", "chengshu_jiejie"],
            ["Helen", "you_pingjing"],
            ["Lore", "calm_story1"],
            ["Crag", "uk_man2"],
            ["Prattle", "laopopo_speech02"],
            ["Hearth", "heainainai_speech02"],
            ["The Reader", "reader_en_m-v1"],
            ["Commercial Lady", "commercial_lady_en_f-v1"]
        ]

        return {
            "required": {
                "video_id": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
                "task_id": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
                "mode": (["text2video", "audio2video"], {"default": "text2video"}),
                "text": ("STRING", {"multiline": True, "default": ""}),
                "voice_language": (["zh", "en"], {"default": "zh"}),
                "zh_voice": ([name for name, _ in cls.zh_voices], {"default": cls.zh_voices[0][0]}),
                "en_voice": ([name for name, _ in cls.en_voices], {"default": cls.en_voices[0][0]}),
                "voice_speed": ("FLOAT", {"default": 1.0, "min": 0.8, "max": 2.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647})
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "video_url": ("STRING", {"default": ""}),
                "audio_type": (["file", "url"], {"default": "file"}),
                "audio_file": ("STRING", {"default": ""}),
                "audio_url": ("STRING", {"default": ""})
            }
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "task_id", "response")
    FUNCTION = "process_lip_sync"
    CATEGORY = CATEGORY_TYPE

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300
        self.zh_voice_map = {name: voice_id for name, voice_id in self.__class__.zh_voices}
        self.en_voice_map = {name: voice_id for name, voice_id in self.__class__.en_voices}

    def process_lip_sync(self, video_id, task_id, mode, text, voice_language, zh_voice, en_voice, voice_speed, seed=0,
                    video_url="", audio_type="file", audio_file="", audio_url="", api_key=""):

        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key

        if not self.api_key:
            error_response = {"task_status": "failed", "task_status_msg": "API key not found in Comflyapi.json"}
            return ("", "", "", json.dumps(error_response))

        if voice_language == "zh":
            voice_id = self.zh_voice_map.get(zh_voice, "")
        else:
            voice_id = self.en_voice_map.get(en_voice, "")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "input": {
                "task_id": task_id,
                "mode": mode,
                "video_id": video_id if video_id else None,
                "video_url": video_url if video_url else None,
                "text": text if mode == "text2video" else None,
                "voice_id": voice_id if mode == "text2video" else None,
                "voice_language": voice_language if mode == "text2video" else None,
                "voice_speed": voice_speed if mode == "text2video" else None,
                "audio_type": audio_type if mode == "audio2video" else None,
                "audio_file": audio_file if mode == "audio2video" and audio_type == "file" else None,
                "audio_url": audio_url if mode == "audio2video" and audio_type == "url" else None,
                "seed": seed
            }
        }
        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(5)
            response = requests.post(
                f"{get_baseurl()}/kling/v1/videos/lip-sync",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            if result["code"] != 0:
                error_response = {"task_status": "failed", "task_status_msg": f"API Error: {result['message']}"}
                return ("", "", "", json.dumps(error_response))

            task_id = result["data"]["task_id"]
            pbar.update_absolute(10)

            last_status = {}
            while True:
                time.sleep(2)
                status_response = requests.get(
                    f"{get_baseurl()}/kling/v1/videos/lip-sync/{task_id}",
                    headers=headers,
                    timeout=self.timeout
                )
                status_response.raise_for_status()
                status_result = status_response.json()
                last_status = status_result["data"]

                if status_result["data"]["task_status"] == "processing":
                    progress = status_result["data"].get("progress", 50)
                    pbar.update_absolute(progress)
                elif status_result["data"]["task_status"] == "succeed":
                    pbar.update_absolute(100)
                    video_url = status_result["data"]["task_result"]["videos"][0]["url"]

                    response_data = {
                        "task_status": "succeed",
                        "task_status_msg": "Lip sync completed successfully",
                        "progress": 100,
                        "video_url": video_url
                    }

                    video_adapter = ComflyVideoAdapter(video_url)
                    return (video_adapter, video_url, task_id, json.dumps(response_data))

                elif status_result["data"]["task_status"] == "failed":
                    error_msg = status_result["data"].get("task_status_msg", "Unknown error")
                    error_response = {
                        "task_status": "failed",
                        "task_status_msg": error_msg,
                    }
                    print(f"Task failed: {error_msg}")
                    return ("", "", task_id, json.dumps(error_response))

        except Exception as e:
            error_response = {"task_status": "failed", "task_status_msg": f"Error in lip sync process: {str(e)}"}
            print(f"Error in lip sync process: {str(e)}")
            return ("", "", "", json.dumps(error_response))


# --- Midjourney Nodes ---
class NakuNodeAPIBaseNode:
    def __init__(self):
        self.midjourney_api_url = {
            "turbo mode": f"{get_baseurl()}/mj-turbo",
            "fast mode": f"{get_baseurl()}/mj-fast",
            "relax mode": f"{get_baseurl()}/mj-relax"
        }
        self.api_key = get_config().get('api_key', '')
        self.speed = "fast mode"
        self.timeout = 800

    def set_speed(self, speed):
        self.speed = speed

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key
        }

    def generateUUID(self):
        return str(uuid.uuid4())

    async def midjourney_submit_action(self, action, taskId, index, custom_id):
        headers = self.get_headers()
        payload = {
            "customId": custom_id,
            "taskId": taskId
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.midjourney_api_url[self.speed]}/mj/submit/action", headers=headers, json=payload, timeout=self.timeout) as response:
                    if response.status == 200:
                        try:
                            data = await response.json()

                            if data.get("status") == "FAILURE":
                                fail_reason = data.get("fail_reason", "Unknown failure reason")
                                error_message = f"Action submission failed: {fail_reason}"
                                print(error_message)
                                raise Exception(error_message)

                            return data
                        except aiohttp.client_exceptions.ContentTypeError:
                            text_response = await response.text()

                            try:
                                import json
                                data = json.loads(text_response)

                                if data.get("status") == "FAILURE":
                                    fail_reason = data.get("fail_reason", "Unknown failure reason")
                                    error_message = f"Action submission failed: {fail_reason}"
                                    print(error_message)
                                    raise Exception(error_message)

                                return data
                            except json.JSONDecodeError:
                                if text_response and len(text_response) < 100:
                                    return {"result": text_response.strip()}
                                raise Exception(f"Invalid response format: {text_response}")
                    else:
                        error_message = f"Error submitting Midjourney action: {response.status}"
                        try:
                            error_details = await response.text()
                            error_message += f" - {error_details}"
                        except:
                            pass
                        raise Exception(error_message)
        except asyncio.TimeoutError:
            error_message = f"Timeout error: Request to submit action timed out after {self.timeout} seconds"
            raise Exception(error_message)
        except Exception as e:
            if "Action submission failed" in str(e):
                raise
            print(f"Exception in midjourney_submit_action: {str(e)}")
            raise e


class NakuNodeAPI_upload(NakuNodeAPIBaseNode):
    """
    NakuNodeAPI_upload node
    Uploads an image to Midjourney and returns the URL link.
    Inputs:
        image (IMAGE): Input image to be uploaded.
    Outputs:
        url (STRING): URL link of the uploaded image.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "api_key": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("url",)
    FUNCTION = "upload_image"
    CATEGORY = CATEGORY_TYPE

    async def upload_image_to_midjourney(self, image):

        image = tensor2pil_naku(image)[0]
        buffered = BytesIO()
        image_format = "PNG"
        image.save(buffered, format=image_format)
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        payload = {
            "base64Array": [f"data:image/{image_format.lower()};base64,{image_base64}"],
            "instanceId": "",
            "notifyHook": ""
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.midjourney_api_url[self.speed]}/mj/submit/upload-discord-images", headers=self.get_headers(), json=payload, timeout=self.timeout) as response:
                    if response.status == 200:
                        try:
                            data = await response.json()

                            if data.get("status") == "FAILURE":
                                fail_reason = data.get("fail_reason", "Unknown failure reason")
                                error_message = f"Image upload failed: {fail_reason}"
                                print(error_message)
                                raise Exception(error_message)

                            if "result" in data and data["result"]:
                                return data["result"][0]
                            else:
                                error_message = f"Unexpected response from Midjourney API: {data}"
                                raise Exception(error_message)
                        except aiohttp.client_exceptions.ContentTypeError:
                            text_response = await response.text()
                            try:
                                import json
                                data = json.loads(text_response)
                                if data.get("status") == "FAILURE":
                                    fail_reason = data.get("fail_reason", "Unknown failure reason")
                                    error_message = f"Image upload failed: {fail_reason}"
                                    print(error_message)
                                    raise Exception(error_message)

                                if "result" in data and data["result"]:
                                    return data["result"][0]
                            except (json.JSONDecodeError, KeyError):
                                if text_response and len(text_response) < 100:
                                    return text_response.strip()
                                raise Exception(f"Invalid response format: {text_response}")
                    else:
                        error_message = f"Error uploading image to Midjourney: {response.status}"
                        try:
                            error_details = await response.text()
                            error_message += f" - {error_details}"
                        except:
                            pass
                        raise Exception(error_message)

        except asyncio.TimeoutError:
            error_message = f"Timeout error: Request to upload image timed out after {self.timeout} seconds"
            raise Exception(error_message)

    def upload_image(self, image, api_key=""):
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key

        try:
            pil_image = tensor2pil_naku(image)[0]
            buffered = BytesIO()
            pil_image.save(buffered, format="PNG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            payload = {
                "base64Array": [f"data:image/png;base64,{image_base64}"],
                "instanceId": "",
                "notifyHook": ""
            }

            response = requests.post(
                f"{self.midjourney_api_url[self.speed]}/mj/submit/upload-discord-images",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "FAILURE":
                    fail_reason = result.get("fail_reason", "Unknown failure reason")
                    error_message = f"Image upload failed: {fail_reason}"
                    print(error_message)
                    raise Exception(error_message)

                if "result" in result and result["result"]:
                    return (result["result"][0],)
                else:
                    error_message = f"Unexpected response from Midjourney API: {result}"
                    raise Exception(error_message)
            else:
                error_message = f"Error uploading image to Midjourney: {response.status_code} - {response.text}"
                raise Exception(error_message)

        except Exception as e:
            print(f"Error in upload_image: {str(e)}")
            raise e


class NakuNodeAPI_Mj(NakuNodeAPIBaseNode):
    """
    NakuNodeAPI_Mj node
    Processes text or image inputs using Midjourney AI model and returns the processed results.
    Inputs:
        text (STRING, optional): Input text.
        api_key (STRING): API key for Midjourney.
        model_version (STRING): Selected Midjourney model version (v 6.1, v 6.0, v 5.2, v 5.1, niji 6, niji 5, niji 4).
        speed (STRING): Selected speed mode (turbo mode, fast mode, relax mode).
        ar (STRING): Aspect ratio.
        no (STRING): Number of images.
        c (STRING): Chaos value (0-100).
        s (STRING): Stylize value (0-1000).
        iw (STRING): Image weight (0-2).
        tile (BOOL): Enable/disable tile.
        r (STRING): Repeat value (1-40).
        video (BOOL): Enable/disable video.
        sw (STRING): Style weight (0-1000).
        cw (STRING): Color weight (0-100).
        sv (STRING): Style variation (1-4).
        seed (INT): Random seed.
        cref (STRING): Creative reference.
        oref (STRING): Object reference.
        sref (STRING): Style reference.
        positive (STRING): Additional positive prompt to be appended to the main prompt.
    Outputs:
        image_url (STRING): URL of the processed image.
        text (STRING): Processed text output.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "speed": (["turbo mode", "fast mode", "relax mode"], {"default": "fast mode"}),
            },
            "optional": {
                "text_en": ("STRING", {"multiline": True, "default": ""}),
                "api_key": ("STRING", {"default": ""}),
                "ar": ("STRING", {"default": "1:1"}),
                "model_version": (["v 7", "v 6.1", "v 6.0", "v 5.2", "v 5.1", "niji 6", "niji 5", "niji 4"], {"default": "v 6.1"}),
                "no": ("STRING", {"default": "", "forceInput": True}),
                "c": ("INT", {"default": 0, "min": 0, "max": 100, "forceInput": True}),
                "s": ("INT", {"default": 0, "min": 0, "max": 1000, "forceInput": True}),
                "iw": ("FLOAT", {"default": 0, "min": 0, "max": 2, "forceInput": True}),
                "r": ("INT", {"default": 1, "min": 1, "max": 40, "forceInput": True}),
                "sw": ("INT", {"default": 0, "min": 0, "max": 1000, "forceInput": True}),
                "cw": ("INT", {"default": 0, "min": 0, "max": 100, "forceInput": True}),
                "sv": (["1", "2", "3", "4"], {"default": "1", "forceInput": True}),
                "oref": ("STRING", {"default": "none", "forceInput": True}),
                "cref": ("STRING", {"default": "none", "forceInput": True}),
                "sref": ("STRING", {"default": "none", "forceInput": True}),
                "positive": ("STRING", {"default": "", "forceInput": True}),
                "video": ("BOOLEAN", {"default": False}),
                "tile": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "text", "taskId")
    OUTPUT_NODE = True

    FUNCTION = "process_input"

    CATEGORY = CATEGORY_TYPE

    def __init__(self):
        super().__init__()
        self.image = None
        self.text = ""

    def process_input(self, speed, text, text_en="", image=None, model_version=None, ar=None, no=None, c=None, s=None, iw=None, r=None, sw=None, cw=None, sv=None, video=False, tile=False, seed=0, cref="none", oref="none", sref="none", positive="", api_key=""):
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key

        self.image = image
        self.speed = speed

        prompt = text_en if text_en else text

        if positive:
            prompt += f" {positive}"

        if model_version:
            prompt += f" --{model_version}"
        if ar:
            prompt += f" --ar {ar}"
        if no:
            prompt += f" --no {no}"
        if c:
            prompt += f" --c {c}"
        if s:
            prompt += f" --s {s}"
        if iw:
            prompt += f" --iw {iw}"
        if r:
            prompt += f" --r {r}"
        if sw:
            prompt += f" --sw {sw}"
        if cw:
            prompt += f" --cw {cw}"
        if sv:
            prompt += f" --sv {sv}"
        if video:
            prompt += " --video"
        if tile:
            prompt += " --tile"
        if oref != "none":
            prompt += f" --oref {oref}"
        if cref != "none":
            prompt += f" --cref {cref}"
        if sref != "none":
            prompt += f" --sref {sref}"

        self.text = prompt

        if self.image is not None:
            image_url, text = self.process_image()
        elif self.text:
            pbar = comfy.utils.ProgressBar(10)
            image_url, text, taskId = self.process_text(pbar, ar, no, c, s, iw, tile, r, video, sw, cw, sv, seed)

            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            tensor_image = pil2tensor_naku(image)
            return tensor_image, text, taskId
        else:
            raise ValueError("Either image or text input must be provided for Midjourney model.")

        return image_url, text, taskId

    def process_text_midjourney_sync(self, text, pbar, ar, no, c, s, iw, tile, r, video, sw, cw, sv, seed):
        try:
            taskId = self.midjourney_submit_imagine_task_sync(text, ar, no, c, s, iw, tile, r, video, sw, cw, sv, seed)
            if taskId:
                result = self.poll_task_result_sync(taskId)
                if result and result.get('status') == 'SUCCESS':
                    pbar.update_absolute(100)
                    image_url = result.get('imageUrl', '')
                    return image_url, text, taskId
                else:
                    error_msg = result.get('fail_reason', 'Unknown error') if result else 'Unknown error'
                    print(f"Midjourney task failed: {error_msg}")
                    return "", f"Error: {error_msg}", taskId
            else:
                print("Failed to submit Midjourney task")
                return "", "Error: Failed to submit task", ""
        except Exception as e:
            print(f"Error in process_text_midjourney_sync: {str(e)}")
            return "", f"Error: {str(e)}", ""

    def midjourney_submit_imagine_task_sync(self, prompt, ar, no, c, s, iw, tile, r, video, sw, cw, sv, seed):
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key
        }
        payload = {
            "base64Array": [],
            "instanceId": "",
            "modes": [],
            "notifyHook": "",
            "prompt": prompt,
            "remix": True,
            "state": "",
            "ar": ar,
            "no": no,
            "c": c,
            "s": s,
            "iw": iw,
            "tile": tile,
            "r": r,
            "video": video,
            "sw": sw,
            "cw": cw,
            "sv": sv,
            "seed": seed
        }

        try:
            response = requests.post(
                f"{self.midjourney_api_url[self.speed]}/mj/submit/imagine",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "FAILURE":
                    fail_reason = result.get("fail_reason", "Unknown failure reason")
                    error_message = f"Midjourney task failed: {fail_reason}"
                    print(error_message)
                    raise Exception(error_message)

                return result["result"]
            else:
                error_message = f"Error submitting Midjourney task: {response.status_code} - {response.text}"
                print(error_message)
                raise Exception(error_message)
        except Exception as e:
            print(f"Exception in midjourney_submit_imagine_task_sync: {str(e)}")
            raise e

    def poll_task_result_sync(self, taskId):
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key
        }

        max_retries = 120
        retry_count = 0

        while retry_count < max_retries:
            try:
                response = requests.get(
                    f"{self.midjourney_api_url[self.speed]}/mj/task/{taskId}/fetch",
                    headers=headers,
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    result = response.json()
                    status = result.get("status")

                    if status == "SUCCESS":
                        return result
                    elif status == "FAILURE":
                        return result
                    else:
                        # Update progress based on attempt count
                        progress = min(90, 10 + (retry_count * 80 // max_retries))
                        print(f"Task {taskId} progress: {result.get('progress', 'Unknown')}, status: {status}")
                else:
                    print(f"Error polling task result: {response.status_code}")

            except Exception as e:
                print(f"Exception polling task result: {str(e)}")

            retry_count += 1
            time.sleep(5)  # Wait 5 seconds between polls

        return {"status": "FAILURE", "fail_reason": "Timeout waiting for task completion"}


class NakuNodeAPI_Mju(NakuNodeAPIBaseNode):
    class MidjourneyError(Exception):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": (["VARIATION", "UPSCALE", "REROLL"], {"default": "UPSCALE"}),
                "index": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "taskId": ("STRING", {"default": ""}),
                "speed": (["turbo mode", "fast mode", "relax mode"], {"default": "fast mode"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "customId": ("STRING", {"default": ""})
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("image_url", "message", "taskId")
    FUNCTION = "process"
    CATEGORY = CATEGORY_TYPE

    def process(self, action, index, taskId, speed, api_key="", customId=""):
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key

        self.speed = speed

        try:
            result = self.midjourney_submit_action_sync(action, taskId, index, customId)
            taskId = result["result"]
            message = f"Action submitted successfully. Task ID: {taskId}"

            return ("", message, taskId)
        except Exception as e:
            error_message = f"Error processing action: {str(e)}"
            print(error_message)
            return ("", error_message, "")

    def midjourney_submit_action_sync(self, action, taskId, index, custom_id):
        headers = self.get_headers()
        payload = {
            "customId": custom_id,
            "taskId": taskId
        }

        response = requests.post(
            f"{self.midjourney_api_url[self.speed]}/mj/submit/action",
            headers=headers,
            json=payload,
            timeout=self.timeout
        )

        if response.status_code == 200:
            result = response.json()

            if result.get("status") == "FAILURE":
                fail_reason = result.get("fail_reason", "Unknown failure reason")
                error_message = f"Action submission failed: {fail_reason}"
                print(error_message)
                raise Exception(error_message)

            return result
        else:
            error_message = f"Error submitting Midjourney action: {response.status_code}"
            print(error_message)
            raise Exception(error_message)


class NakuNodeAPI_Mjv(NakuNodeAPIBaseNode):
    class MidjourneyError(Exception):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": (["DESCRIBE", "BLEND"], {"default": "DESCRIBE"}),
                "speed": (["turbo mode", "fast mode", "relax mode"], {"default": "fast mode"}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "api_key": ("STRING", {"default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": ""})
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("content", "message", "taskId")
    FUNCTION = "process"
    CATEGORY = CATEGORY_TYPE

    def process(self, action, speed, image1=None, image2=None, image3=None, image4=None, api_key="", prompt=""):
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key

        self.speed = speed

        try:
            image_urls = []
            for img in [image1, image2, image3, image4]:
                if img is not None:
                    url = self.upload_image_sync(img)
                    if url:
                        image_urls.append(url)

            if action == "DESCRIBE":
                result = self.midjourney_describe_sync(image_urls)
            elif action == "BLEND":
                result = self.midjourney_blend_sync(image_urls, prompt)
            else:
                raise ValueError(f"Unsupported action: {action}")

            taskId = result.get("result", "")
            content = result.get("content", "")
            message = f"Action submitted successfully. Task ID: {taskId}"

            return (content, message, taskId)
        except Exception as e:
            error_message = f"Error processing action: {str(e)}"
            print(error_message)
            return ("", error_message, "")

    def upload_image_sync(self, image_tensor):
        # Convert tensor to PIL image
        pil_image = tensor2pil_naku(image_tensor)[0]
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        payload = {
            "base64Array": [f"data:image/png;base64,{image_base64}"],
            "instanceId": "",
            "notifyHook": ""
        }

        response = requests.post(
            f"{self.midjourney_api_url[self.speed]}/mj/submit/upload-discord-images",
            headers=self.get_headers(),
            json=payload,
            timeout=self.timeout
        )

        if response.status_code == 200:
            result = response.json()
            if result.get("status") == "FAILURE":
                fail_reason = result.get("fail_reason", "Unknown failure reason")
                error_message = f"Image upload failed: {fail_reason}"
                print(error_message)
                raise Exception(error_message)

            if "result" in result and result["result"]:
                return result["result"][0]
            else:
                error_message = f"Unexpected response from Midjourney API: {result}"
                raise Exception(error_message)
        else:
            error_message = f"Error uploading image to Midjourney: {response.status_code} - {response.text}"
            raise Exception(error_message)

    def midjourney_describe_sync(self, image_urls):
        headers = self.get_headers()
        payload = {
            "base64Array": image_urls[:1],  # Describe action typically takes one image
            "instanceId": "",
            "notifyHook": ""
        }

        response = requests.post(
            f"{self.midjourney_api_url[self.speed]}/mj/submit/describe",
            headers=headers,
            json=payload,
            timeout=self.timeout
        )

        if response.status_code == 200:
            result = response.json()
            if result.get("status") == "FAILURE":
                fail_reason = result.get("fail_reason", "Unknown failure reason")
                error_message = f"Describe action failed: {fail_reason}"
                print(error_message)
                raise Exception(error_message)

            return result
        else:
            error_message = f"Error submitting describe action: {response.status_code} - {response.text}"
            raise Exception(error_message)

    def midjourney_blend_sync(self, image_urls, prompt):
        headers = self.get_headers()
        payload = {
            "base64Array": image_urls,
            "prompt": prompt,
            "instanceId": "",
            "notifyHook": ""
        }

        response = requests.post(
            f"{self.midjourney_api_url[self.speed]}/mj/submit/blend",
            headers=headers,
            json=payload,
            timeout=self.timeout
        )

        if response.status_code == 200:
            result = response.json()
            if result.get("status") == "FAILURE":
                fail_reason = result.get("fail_reason", "Unknown failure reason")
                error_message = f"Blend action failed: {fail_reason}"
                print(error_message)
                raise Exception(error_message)

            return result
        else:
            error_message = f"Error submitting blend action: {response.status_code} - {response.text}"
            raise Exception(error_message)


class NakuNodeAPI_Mj_swap_face(NakuNodeAPIBaseNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_image": ("IMAGE",),
                "source_image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "speed": (["turbo mode", "fast mode", "relax mode"], {"default": "fast mode"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""})
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("output_image", "message")
    FUNCTION = "swap_face"
    CATEGORY = CATEGORY_TYPE

    def swap_face(self, target_image, source_image, strength, speed, api_key=""):
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key

        self.speed = speed

        try:
            target_url = self.upload_image_sync(target_image)
            source_url = self.upload_image_sync(source_image)

            headers = self.get_headers()
            payload = {
                "target_url": target_url,
                "source_url": source_url,
                "strength": strength
            }

            response = requests.post(
                f"{self.midjourney_api_url[self.speed]}/mj/swap/face",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "FAILURE":
                    fail_reason = result.get("fail_reason", "Unknown failure reason")
                    error_message = f"Face swap failed: {fail_reason}"
                    print(error_message)
                    return (target_image, error_message)

                # Get the result image
                result_url = result.get("result_url", "")
                if result_url:
                    response = requests.get(result_url)
                    if response.status_code == 200:
                        img = Image.open(BytesIO(response.content))
                        output_tensor = pil2tensor_naku(img)
                        return (output_tensor, "Face swap completed successfully")
                    else:
                        return (target_image, f"Failed to download result image: {response.status_code}")
                else:
                    return (target_image, "No result URL returned from face swap API")
            else:
                error_message = f"Error submitting face swap request: {response.status_code} - {response.text}"
                print(error_message)
                return (target_image, error_message)

        except Exception as e:
            error_message = f"Error in face swap: {str(e)}"
            print(error_message)
            return (target_image, error_message)

    def upload_image_sync(self, image_tensor):
        # Convert tensor to PIL image
        pil_image = tensor2pil_naku(image_tensor)[0]
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        payload = {
            "base64Array": [f"data:image/png;base64,{image_base64}"],
            "instanceId": "",
            "notifyHook": ""
        }

        response = requests.post(
            f"{self.midjourney_api_url[self.speed]}/mj/submit/upload-discord-images",
            headers=self.get_headers(),
            json=payload,
            timeout=self.timeout
        )

        if response.status_code == 200:
            result = response.json()
            if result.get("status") == "FAILURE":
                fail_reason = result.get("fail_reason", "Unknown failure reason")
                error_message = f"Image upload failed: {fail_reason}"
                print(error_message)
                raise Exception(error_message)

            if "result" in result and result["result"]:
                return result["result"][0]
            else:
                error_message = f"Unexpected response from Midjourney API: {result}"
                raise Exception(error_message)
        else:
            error_message = f"Error uploading image to Midjourney: {response.status_code} - {response.text}"
            raise Exception(error_message)


class NakuNodeAPI_mj_video(NakuNodeAPIBaseNode):
    """
    NakuNodeAPI_mj_video node
    Generates videos using Midjourney's video API based on text prompts or text+image combination.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (["m1", "m2", "m3", "m4"], {"default": "m1"}),
                "duration": ("INT", {"default": 10, "min": 1, "max": 60}),
                "aspect_ratio": (["16:9", "9:16", "1:1", "4:3"], {"default": "16:9"}),
                "speed": (["turbo mode", "fast mode", "relax mode"], {"default": "fast mode"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "style_reference": ("STRING", {"default": ""}),
                "negative_prompt": ("STRING", {"default": ""}),
                "api_key": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            }
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "message")
    FUNCTION = "generate_video"
    CATEGORY = CATEGORY_TYPE

    def generate_video(self, prompt, model, duration, aspect_ratio, speed, image=None, style_reference="", negative_prompt="", api_key="", seed=0):
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key

        self.speed = speed

        try:
            headers = self.get_headers()
            payload = {
                "prompt": prompt,
                "model": model,
                "duration": duration,
                "aspect_ratio": aspect_ratio,
                "negative_prompt": negative_prompt,
                "seed": seed
            }

            if style_reference:
                payload["style_reference"] = style_reference

            if image is not None:
                # Upload the image
                image_url = self.upload_image_sync(image)
                if image_url:
                    payload["image_url"] = image_url

            response = requests.post(
                f"{self.midjourney_api_url[self.speed]}/mj/video/submit",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "FAILURE":
                    fail_reason = result.get("fail_reason", "Unknown failure reason")
                    error_message = f"Video generation failed: {fail_reason}"
                    print(error_message)
                    return ("", "", error_message)

                task_id = result.get("task_id", "")
                if task_id:
                    # Wait for video generation to complete
                    video_url = self.wait_for_video_completion(task_id)
                    if video_url:
                        video_adapter = ComflyVideoAdapter(video_url)
                        return (video_adapter, video_url, f"Video generated successfully. Task ID: {task_id}")
                    else:
                        return ("", "", f"Failed to get video URL for task: {task_id}")
                else:
                    return ("", "", "No task ID returned from video generation API")
            else:
                error_message = f"Error submitting video generation request: {response.status_code} - {response.text}"
                print(error_message)
                return ("", "", error_message)

        except Exception as e:
            error_message = f"Error in video generation: {str(e)}"
            print(error_message)
            return ("", "", error_message)

    def wait_for_video_completion(self, task_id):
        max_retries = 120  # Wait up to 10 minutes (120 * 5 seconds)
        retry_count = 0

        while retry_count < max_retries:
            try:
                headers = self.get_headers()
                response = requests.get(
                    f"{self.midjourney_api_url[self.speed]}/mj/video/status/{task_id}",
                    headers=headers,
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    result = response.json()
                    status = result.get("status", "")

                    if status == "completed":
                        return result.get("video_url", "")
                    elif status == "failed":
                        print(f"Video generation failed for task {task_id}")
                        return ""
                    else:
                        # Still processing, wait and retry
                        time.sleep(5)
                        retry_count += 1
                else:
                    print(f"Error checking video status: {response.status_code}")
                    time.sleep(5)
                    retry_count += 1

            except Exception as e:
                print(f"Exception checking video status: {str(e)}")
                time.sleep(5)
                retry_count += 1

        print(f"Timeout waiting for video completion for task {task_id}")
        return ""

    def upload_image_sync(self, image_tensor):
        # Convert tensor to PIL image
        pil_image = tensor2pil_naku(image_tensor)[0]
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        payload = {
            "base64Array": [f"data:image/png;base64,{image_base64}"],
            "instanceId": "",
            "notifyHook": ""
        }

        response = requests.post(
            f"{self.midjourney_api_url[self.speed]}/mj/submit/upload-discord-images",
            headers=self.get_headers(),
            json=payload,
            timeout=self.timeout
        )

        if response.status_code == 200:
            result = response.json()
            if result.get("status") == "FAILURE":
                fail_reason = result.get("fail_reason", "Unknown failure reason")
                error_message = f"Image upload failed: {fail_reason}"
                print(error_message)
                raise Exception(error_message)

            if "result" in result and result["result"]:
                return result["result"][0]
            else:
                error_message = f"Unexpected response from Midjourney API: {result}"
                raise Exception(error_message)
        else:
            error_message = f"Error uploading image to Midjourney: {response.status_code} - {response.text}"
            raise Exception(error_message)


class NakuNodeAPI_mj_video_extend(NakuNodeAPIBaseNode):
    """
    NakuNodeAPI_mj_video_extend node
    Extends a Midjourney video based on a task ID.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task_id": ("STRING", {"default": "", "multiline": False}),
                "extension_duration": ("INT", {"default": 5, "min": 1, "max": 30}),
                "speed": (["turbo mode", "fast mode", "relax mode"], {"default": "fast mode"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""})
            }
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "message")
    FUNCTION = "extend_video"
    CATEGORY = CATEGORY_TYPE

    def extend_video(self, task_id, extension_duration, speed, api_key=""):
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key

        self.speed = speed

        try:
            headers = self.get_headers()
            payload = {
                "task_id": task_id,
                "extension_duration": extension_duration
            }

            response = requests.post(
                f"{self.midjourney_api_url[self.speed]}/mj/video/extend",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "FAILURE":
                    fail_reason = result.get("fail_reason", "Unknown failure reason")
                    error_message = f"Video extension failed: {fail_reason}"
                    print(error_message)
                    return ("", "", error_message)

                new_task_id = result.get("new_task_id", "")
                if new_task_id:
                    # Wait for video extension to complete
                    video_url = self.wait_for_video_completion(new_task_id)
                    if video_url:
                        video_adapter = ComflyVideoAdapter(video_url)
                        return (video_adapter, video_url, f"Video extended successfully. New Task ID: {new_task_id}")
                    else:
                        return ("", "", f"Failed to get extended video URL for task: {new_task_id}")
                else:
                    return ("", "", "No new task ID returned from video extension API")
            else:
                error_message = f"Error submitting video extension request: {response.status_code} - {response.text}"
                print(error_message)
                return ("", "", error_message)

        except Exception as e:
            error_message = f"Error in video extension: {str(e)}"
            print(error_message)
            return ("", "", error_message)

    def wait_for_video_completion(self, task_id):
        max_retries = 120  # Wait up to 10 minutes (120 * 5 seconds)
        retry_count = 0

        while retry_count < max_retries:
            try:
                headers = self.get_headers()
                response = requests.get(
                    f"{self.midjourney_api_url[self.speed]}/mj/video/status/{task_id}",
                    headers=headers,
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    result = response.json()
                    status = result.get("status", "")

                    if status == "completed":
                        return result.get("video_url", "")
                    elif status == "failed":
                        print(f"Video extension failed for task {task_id}")
                        return ""
                    else:
                        # Still processing, wait and retry
                        time.sleep(5)
                        retry_count += 1
                else:
                    print(f"Error checking video status: {response.status_code}")
                    time.sleep(5)
                    retry_count += 1

            except Exception as e:
                print(f"Exception checking video status: {str(e)}")
                time.sleep(5)
                retry_count += 1

        print(f"Timeout waiting for video completion for task {task_id}")
        return ""


# --- OpenAI Nodes ---
class NakuNodeAPI_gpt_image_edit:

    _last_edited_image = None
    _conversation_history = []

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
            },
            "optional": {
                "mask": ("MASK",),
                "api_key": ("STRING", {"default": ""}),
                "model": (["gpt-image-1", "gpt-image-1.5"], {"default": "gpt-image-1"}),
                "n": ("INT", {"default": 1, "min": 1, "max": 10}),
                "quality": (["auto", "high", "medium", "low"], {"default": "auto"}),
                "size": (["auto", "1024x1024", "1536x1024", "1024x1536"], {"default": "auto"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "clear_chats": ("BOOLEAN", {"default": True}),
                "background": (["auto", "transparent", "opaque"], {"default": "auto"}),
                "output_compression": ("INT", {"default": 100, "min": 0, "max": 100}),
                "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
                "max_retries": ("INT", {"default": 5, "min": 1, "max": 10}),
                "initial_timeout": ("INT", {"default": 900, "min": 60, "max": 1200}),
                "input_fidelity": (["low", "high"], {"default": "low"}),
                "partial_images": ([0, 1, 2, 3], {"default": 0}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("edited_image", "response", "chats")
    FUNCTION = "edit_image"
    CATEGORY = CATEGORY_TYPE

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 900
        self.session = requests.Session()
        retry_strategy = requests.packages.urllib3.util.retry.Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def edit_image(self, image, prompt, mask=None, api_key="", model="gpt-image-1", n=1, quality="auto",
                   size="auto", seed=0, clear_chats=True, background="auto", output_compression=100,
                   output_format="png", max_retries=5, initial_timeout=900, input_fidelity="low", partial_images=0):
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key

        if not self.api_key:
            error_message = "API key not found in Comflyapi.json"
            print(error_message)
            return (image, error_message, self.format_conversation_history())

        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            # Convert image tensor to base64
            pil_image = tensor2pil_naku(image)[0]
            img_buffer = BytesIO()
            pil_image.save(img_buffer, format="PNG")
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

            payload = {
                "model": model,
                "prompt": prompt,
                "n": n,
                "quality": quality,
                "size": size,
                "seed": seed,
                "response_format": "b64_json"
            }

            if mask is not None:
                # Convert mask to base64 if provided
                mask_np = mask.cpu().numpy()
                mask_img = Image.fromarray((mask_np * 255).astype(np.uint8), mode="L")
                mask_buffer = BytesIO()
                mask_img.save(mask_buffer, format="PNG")
                mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode('utf-8')
                payload["mask"] = mask_base64

            if background != "auto":
                payload["background"] = background
            if output_compression != 100:
                payload["output_compression"] = output_compression
            if output_format != "png":
                payload["output_format"] = output_format
            if input_fidelity != "low":
                payload["input_fidelity"] = input_fidelity

            headers = self.get_headers()
            response = requests.post(
                f"{get_baseurl()}/v1/images/edits",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )

            pbar.update_absolute(50)

            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(error_message)
                return (image, error_message, self.format_conversation_history())

            result = response.json()
            pbar.update_absolute(80)

            if "data" not in result or not result["data"]:
                error_message = "No image data in response"
                print(error_message)
                return (image, error_message, self.format_conversation_history())

            # Process the result
            generated_images = []
            for item in result["data"]:
                if "b64_json" in item:
                    img_data = base64.b64decode(item["b64_json"])
                    img = Image.open(BytesIO(img_data))
                    tensor_img = pil2tensor_naku(img)
                    generated_images.append(tensor_img)

            if generated_images:
                combined_tensor = torch.cat(generated_images, dim=0)
                pbar.update_absolute(100)

                # Update conversation history
                if not clear_chats:
                    self._conversation_history.append({
                        "role": "user",
                        "content": prompt,
                        "response": result
                    })
                    self._last_edited_image = combined_tensor
                else:
                    self.clear_conversation_history()

                response_text = f"Image edited successfully using {model}\nPrompt: {prompt}\nSize: {size}"
                return (combined_tensor, response_text, self.format_conversation_history())
            else:
                error_message = "No images were successfully processed"
                print(error_message)
                return (image, error_message, self.format_conversation_history())

        except Exception as e:
            error_message = f"Error in image editing: {str(e)}"
            print(error_message)
            return (image, error_message, self.format_conversation_history())

    def format_conversation_history(self):
        if not self._conversation_history:
            return "No conversation history"
        history_str = "Conversation History:\n"
        for i, entry in enumerate(self._conversation_history):
            history_str += f"{i+1}. {entry['role']}: {entry['content'][:50]}...\n"
        return history_str

    def clear_conversation_history(self):
        self._conversation_history = []


class NakuNodeAPI_gpt_image:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (["gpt-image-1"], {"default": "gpt-image-1"}),
                "size": (["256x256", "512x512", "1024x1024"], {"default": "1024x1024"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "n": ("INT", {"default": 1, "min": 1, "max": 10}),
                "quality": (["standard", "hd"], {"default": "standard"}),
                "style": (["natural", "vivid"], {"default": "natural"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("generated_image", "response")
    FUNCTION = "generate_image"
    CATEGORY = CATEGORY_TYPE

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def generate_image(self, prompt, model, size, api_key="", n=1, quality="standard", style="natural", seed=0):
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key

        if not self.api_key:
            error_message = "API key not found in Comflyapi.json"
            print(error_message)
            blank_image = Image.new('RGB', (512, 512), color='white')
            blank_tensor = pil2tensor_naku(blank_image)
            return (blank_tensor, error_message)

        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            payload = {
                "prompt": prompt,
                "model": model,
                "size": size,
                "n": n,
                "response_format": "b64_json"
            }

            if quality != "standard":
                payload["quality"] = quality
            if style != "natural":
                payload["style"] = style
            if seed != 0:
                payload["seed"] = seed

            headers = self.get_headers()
            response = requests.post(
                f"{get_baseurl()}/v1/images/generations",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )

            pbar.update_absolute(50)

            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(error_message)
                blank_image = Image.new('RGB', (512, 512), color='white')
                blank_tensor = pil2tensor_naku(blank_image)
                return (blank_tensor, error_message)

            result = response.json()
            pbar.update_absolute(80)

            if "data" not in result or not result["data"]:
                error_message = "No image data in response"
                print(error_message)
                blank_image = Image.new('RGB', (512, 512), color='white')
                blank_tensor = pil2tensor_naku(blank_image)
                return (blank_tensor, error_message)

            # Process the result
            generated_images = []
            for item in result["data"]:
                if "b64_json" in item:
                    img_data = base64.b64decode(item["b64_json"])
                    img = Image.open(BytesIO(img_data))
                    tensor_img = pil2tensor_naku(img)
                    generated_images.append(tensor_img)

            if generated_images:
                combined_tensor = torch.cat(generated_images, dim=0)
                pbar.update_absolute(100)

                response_text = f"Image generated successfully using {model}\nPrompt: {prompt}\nSize: {size}\nQuality: {quality}\nStyle: {style}"
                return (combined_tensor, response_text)
            else:
                error_message = "No images were successfully processed"
                print(error_message)
                blank_image = Image.new('RGB', (512, 512), color='white')
                blank_tensor = pil2tensor_naku(blank_image)
                return (blank_tensor, error_message)

        except Exception as e:
            error_message = f"Error in image generation: {str(e)}"
            print(error_message)
            blank_image = Image.new('RGB', (512, 512), color='white')
            blank_tensor = pil2tensor_naku(blank_image)
            return (blank_tensor, error_message)


class NakuNodeAPI_ChatGPTApi:
 
    _last_generated_image_urls = ""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"], {"default": "gpt-4o-mini"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "api_key": ("STRING", {"default": ""}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_tokens": ("INT", {"default": 1000, "min": 1, "max": 4096}),
                "response_format": (["text", "json_object"], {"default": "text"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("response", "image_url", "chats")
    FUNCTION = "generate_text"
    CATEGORY = CATEGORY_TYPE

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.conversation_history = []
        self.timeout = 300

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def tensor_to_base64(self, tensor):
        if tensor is None:
            return None
        if tensor.dtype != torch.uint8:
            tensor = (tensor * 255).clamp(0, 255).byte()
        tensor = tensor.cpu()
        if tensor.shape[-1] == 3:
            img = Image.fromarray(tensor.numpy(), 'RGB')
        else:
            img = Image.fromarray(tensor.numpy(), 'RGBA')
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def generate_text(self, prompt, model, temperature=0.7, top_p=1.0, max_tokens=1000, 
                     response_format="text", seed=0, image=None, api_key=""):
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key

        if not self.api_key:
            return ("API key not found in Comflyapi.json", "", self.format_conversation_history())

        try:
            # Prepare messages with potential image
            messages = [{"role": "user", "content": []}]
            
            # Add text content
            messages[0]["content"].append({
                "type": "text",
                "text": prompt
            })

            # Add image if provided
            if image is not None:
                if len(image.shape) == 4:
                    image = image[0]  # Take first image if batched
                img_b64 = self.tensor_to_base64(image)
                if img_b64:
                    messages[0]["content"].append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_b64}"
                        }
                    })

            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
            }

            if response_format == "json_object":
                payload["response_format"] = {"type": "json_object"}

            if seed > 0:
                payload["seed"] = seed

            headers = self.get_headers()
            response = requests.post(
                f"{get_baseurl()}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()

            # Extract response
            text_response = result["choices"][0]["message"]["content"]

            # Extract potential image URLs from response
            image_urls = []
            import re
            url_pattern = r'https?://\S+\.(?:jpg|jpeg|png|gif|webp|bmp)'
            matches = re.findall(url_pattern, text_response)
            if matches:
                image_urls = matches[:4]  # Limit to first 4 URLs
                self._last_generated_image_urls = ", ".join(image_urls)

            # Update conversation history
            self.conversation_history.append({
                "role": "user",
                "content": prompt
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": text_response
            })

            return (text_response, self._last_generated_image_urls, self.format_conversation_history())

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            return (error_msg, "", self.format_conversation_history())

    def format_conversation_history(self):
        if not self.conversation_history:
            return "No conversation history"
        history_str = "Conversation History:\n"
        for i, entry in enumerate(self.conversation_history):
            history_str += f"{i+1}. {entry['role']}: {entry['content'][:100]}...\n"
        return history_str


class NakuNodeAPI_sora2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (["sora-2-turbo", "sora-2-pro", "sora-2-standard"], {"default": "sora-2-turbo"}),
                "aspect_ratio": (["16:9", "9:16", "1:1", "4:3", "3:4"], {"default": "16:9"}),
                "duration": ("INT", {"default": 5, "min": 1, "max": 10}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "quality": (["standard", "high"], {"default": "standard"}),
                "motion_strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
                "style": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "private": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "response")
    FUNCTION = "generate_video"
    CATEGORY = CATEGORY_TYPE

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 600

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def generate_video(self, prompt, model, aspect_ratio, duration, api_key="", 
                      quality="standard", motion_strength=1.0, style="", seed=0, private=True):
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key

        if not self.api_key:
            error_response = {"status": "error", "message": "API key not found in Comflyapi.json"}
            return ("", "", json.dumps(error_response))

        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            payload = {
                "prompt": prompt,
                "model": model,
                "aspect_ratio": aspect_ratio,
                "duration": duration,
                "quality": quality,
                "motion_strength": motion_strength,
                "private": private
            }

            if style:
                payload["style"] = style
            if seed > 0:
                payload["seed"] = seed

            headers = self.get_headers()
            response = requests.post(
                f"{get_baseurl()}/sora2/generate",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )

            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(error_message)
                return ("", "", json.dumps({"status": "error", "message": error_message}))

            result = response.json()
            pbar.update_absolute(30)

            if "task_id" not in result:
                error_message = "No task ID in response"
                print(error_message)
                return ("", "", json.dumps({"status": "error", "message": error_message}))

            task_id = result["task_id"]
            pbar.update_absolute(40)

            # Poll for completion
            max_attempts = 180  # 15 minutes with 5-second intervals
            attempts = 0
            
            while attempts < max_attempts:
                time.sleep(5)
                attempts += 1

                status_response = requests.get(
                    f"{get_baseurl()}/sora2/status/{task_id}",
                    headers=self.get_headers(),
                    timeout=self.timeout
                )

                if status_response.status_code == 200:
                    status_result = status_response.json()
                    status = status_result.get("status", "")

                    progress_value = min(90, 40 + (attempts * 50 // max_attempts))
                    pbar.update_absolute(progress_value)

                    if status == "completed":
                        video_url = status_result.get("video_url", "")
                        if video_url:
                            pbar.update_absolute(100)
                            
                            response_data = {
                                "status": "success",
                                "task_id": task_id,
                                "video_url": video_url,
                                "prompt": prompt,
                                "model": model,
                                "aspect_ratio": aspect_ratio,
                                "duration": duration,
                                "quality": quality
                            }

                            video_adapter = ComflyVideoAdapter(video_url)
                            return (video_adapter, video_url, json.dumps(response_data))
                    elif status == "failed":
                        error_message = status_result.get("error_message", "Unknown error")
                        return ("", "", json.dumps({"status": "error", "message": f"Video generation failed: {error_message}"}))
                    elif status == "processing":
                        continue
                    else:
                        print(f"Unexpected status: {status}")

                pbar.update_absolute(progress_value)

            return ("", "", json.dumps({"status": "error", "message": "Video generation timeout"}))

        except Exception as e:
            error_message = f"Error in video generation: {str(e)}"
            print(error_message)
            return ("", "", json.dumps({"status": "error", "message": error_message}))


class NakuNodeAPI_sora2_character:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "username": ("STRING", {"default": ""}),
                "prompt": ("STRING", {"multiline": True}),
                "model": (["sora-2-turbo", "sora-2-pro", "sora-2-standard"], {"default": "sora-2-turbo"}),
                "aspect_ratio": (["16:9", "9:16", "1:1", "4:3", "3:4"], {"default": "16:9"}),
                "duration": ("INT", {"default": 5, "min": 1, "max": 10}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "quality": (["standard", "high"], {"default": "standard"}),
                "motion_strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "private": ("BOOLEAN", {"default": True}),
                "from_task": ("STRING", {"default": ""}),  # Added the new parameter
            }
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "response")
    FUNCTION = "generate_character_video"
    CATEGORY = CATEGORY_TYPE

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 600

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def generate_character_video(self, username, prompt, model, aspect_ratio, duration, 
                               api_key="", quality="standard", motion_strength=1.0, 
                               seed=0, private=True, from_task=""):
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key

        if not self.api_key:
            error_response = {"status": "error", "message": "API key not found in Comflyapi.json"}
            return ("", "", json.dumps(error_response))

        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            # Include username in the prompt
            full_prompt = f"{prompt} @username: {username}"
            
            payload = {
                "prompt": full_prompt,
                "model": model,
                "aspect_ratio": aspect_ratio,
                "duration": duration,
                "quality": quality,
                "motion_strength": motion_strength,
                "private": private
            }

            if seed > 0:
                payload["seed"] = seed
            if from_task:  # Include from_task if provided
                payload["from_task"] = from_task

            headers = self.get_headers()
            response = requests.post(
                f"{get_baseurl()}/sora2/character/generate",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )

            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(error_message)
                return ("", "", json.dumps({"status": "error", "message": error_message}))

            result = response.json()
            pbar.update_absolute(30)

            if "task_id" not in result:
                error_message = "No task ID in response"
                print(error_message)
                return ("", "", json.dumps({"status": "error", "message": error_message}))

            task_id = result["task_id"]
            pbar.update_absolute(40)

            # Poll for completion
            max_attempts = 180  # 15 minutes with 5-second intervals
            attempts = 0
            
            while attempts < max_attempts:
                time.sleep(5)
                attempts += 1

                status_response = requests.get(
                    f"{get_baseurl()}/sora2/character/status/{task_id}",
                    headers=self.get_headers(),
                    timeout=self.timeout
                )

                if status_response.status_code == 200:
                    status_result = status_response.json()
                    status = status_result.get("status", "")

                    progress_value = min(90, 40 + (attempts * 50 // max_attempts))
                    pbar.update_absolute(progress_value)

                    if status == "completed":
                        video_url = status_result.get("video_url", "")
                        if video_url:
                            pbar.update_absolute(100)
                            
                            response_data = {
                                "status": "success",
                                "task_id": task_id,
                                "video_url": video_url,
                                "username": username,
                                "prompt": prompt,
                                "model": model,
                                "aspect_ratio": aspect_ratio,
                                "duration": duration,
                                "quality": quality
                            }

                            video_adapter = ComflyVideoAdapter(video_url)
                            return (video_adapter, video_url, json.dumps(response_data))
                    elif status == "failed":
                        error_message = status_result.get("error_message", "Unknown error")
                        return ("", "", json.dumps({"status": "error", "message": f"Character video generation failed: {error_message}"}))
                    elif status == "processing":
                        continue
                    else:
                        print(f"Unexpected status: {status}")

                pbar.update_absolute(progress_value)

            return ("", "", json.dumps({"status": "error", "message": "Character video generation timeout"}))

        except Exception as e:
            error_message = f"Error in character video generation: {str(e)}"
            print(error_message)
            return ("", "", json.dumps({"status": "error", "message": error_message}))


# --- Vidu Nodes ---
class NakuNodeAPI_vidu_img2video:
    VOICE_OPTIONS = {
        "中文(普通话)": [
            "male-qn-qingse", "male-qn-jingying", "male-qn-badao", "male-qn-daxuesheng",
            "female-shaonv", "female-yujie", "female-chengshu", "female-tianmei",
            "male-qn-qingse-jingpin", "male-qn-jingying-jingpin", "male-qn-badao-jingpin",
            "male-qn-daxuesheng-jingpin", "female-shaonv-jingpin", "female-yujie-jingpin",
            "female-chengshu-jingpin", "female-tianmei-jingpin",
            "clever_boy", "cute_boy", "lovely_girl", "cartoon_pig",
            "bingjiao_didi", "junlang_nanyou", "chunzhen_xuedi", "lengdan_xiongzhang", "badao_shaoye",
            "tianxin_xiaoling", "qiaopi_mengmei", "wumei_yujie", "diadia_xuemei", "danya_xuejie",
            "Chinese (Mandarin)_Reliable_Executive", "Chinese (Mandarin)_News_Anchor",
            "Chinese (Mandarin)_Mature_Woman", "Chinese (Mandarin)_Unrestrained_Young_Man",
            "Arrogant_Miss", "Robot_Armor", "Chinese (Mandarin)_Kind-hearted_Antie",
            "Chinese (Mandarin)_HK_Flight_Attendant", "Chinese (Mandarin)_Humorous_Elder",
            "Chinese (Mandarin)_Gentleman", "Chinese (Mandarin)_Warm_Bestie",
            "Chinese (Mandarin)_Male_Announcer", "Chinese (Mandarin)_Sweet_Lady",
            "Chinese (Mandarin)_Southern_Young_Man", "Chinese (Mandarin)_Wise_Women",
            "Chinese (Mandarin)_Gentle_Youth", "Chinese (Mandarin)_Warm_Girl",
            "Chinese (Mandarin)_Kind-hearted_Elder", "Chinese (Mandarin)_Cute_Spirit",
            "Chinese (Mandarin)_Radio_Host", "Chinese (Mandarin)_Lyrical_Voice",
            "Chinese (Mandarin)_Straightforward_Boy", "Chinese (Mandarin)_Sincere_Adult",
            "Chinese (Mandarin)_Gentle_Senior", "Chinese (Mandarin)_Stubborn_Friend",
            "Chinese (Mandarin)_Crisp_Girl", "Chinese (Mandarin)_Pure-hearted_Boy",
            "Chinese (Mandarin)_Soft_Girl"
        ],
        "中文(粤语)": [
            "Cantonese_ProfessionalHost（F)", "Cantonese_GentleLady",
            "Cantonese_ProfessionalHost（M)", "Cantonese_PlayfulMan",
            "Cantonese_CuteGirl", "Cantonese_KindWoman"
        ],
        "English": [
            "Santa_Claus", "Grinch", "Rudolph", "Arnold", "Charming_Santa", "Charming_Lady",
            "Sweet_Girl", "Cute_Elf", "Attractive_Girl", "Serene_Woman",
            "English_Trustworthy_Man", "English_Graceful_Lady", "English_Aussie_Bloke",
            "English_Whispering_girl", "English_Diligent_Man", "English_Gentle-voiced_man"
        ],
        "日本語": [
            "Japanese_IntellectualSenior", "Japanese_DecisivePrincess", "Japanese_LoyalKnight",
            "Japanese_DominantMan", "Japanese_SeriousCommander", "Japanese_ColdQueen",
            "Japanese_DependableWoman", "Japanese_GentleButler", "Japanese_KindLady",
            "Japanese_CalmLady", "Japanese_OptimisticYouth", "Japanese_GenerousIzakayaOwner",
            "Japanese_SportyStudent", "Japanese_InnocentBoy", "Japanese_GracefulMaiden"
        ],
        "한국어": [
            "Korean_SweetGirl", "Korean_CheerfulBoyfriend", "Korean_EnchantingSister",
            "Korean_ShyGirl", "Korean_ReliableSister", "Korean_StrictBoss", "Korean_SassyGirl",
            "Korean_ChildhoodFriendGirl", "Korean_PlayboyCharmer", "Korean_ElegantPrincess",
            "Korean_BraveFemaleWarrior", "Korean_BraveYouth", "Korean_CalmLady",
            "Korean_EnthusiasticTeen", "Korean_SoothingLady", "Korean_IntellectualSenior",
            "Korean_LonelyWarrior", "Korean_MatureLady", "Korean_InnocentBoy",
            "Korean_CharmingSister", "Korean_AthleticStudent", "Korean_BraveAdventurer",
            "Korean_CalmGentleman", "Korean_WiseElf", "Korean_CheerfulCoolJunior",
            "Korean_DecisiveQueen", "Korean_ColdYoungMan", "Korean_MysteriousGirl",
            "Korean_QuirkyGirl", "Korean_ConsiderateSenior", "Korean_CheerfulLittleSister",
            "Korean_DominantMan", "Korean_AirheadedGirl", "Korean_ReliableYouth",
            "Korean_FriendlyBigSister", "Korean_GentleBoss", "Korean_ColdGirl",
            "Korean_HaughtyLady", "Korean_CharmingElderSister", "Korean_IntellectualMan",
            "Korean_CaringWoman", "Korean_WiseTeacher", "Korean_ConfidentBoss",
            "Korean_AthleticGirl", "Korean_PossessiveMan", "Korean_GentleWoman",
            "Korean_CockyGuy", "Korean_ThoughtfulWoman", "Korean_OptimisticYouth"
        ],
        "Español": [
            "Spanish_SereneWoman", "Spanish_MaturePartner", "Spanish_CaptivatingStoryteller",
            "Spanish_Narrator", "Spanish_WiseScholar", "Spanish_Kind-heartedGirl",
            "Spanish_DeterminedManager", "Spanish_BossyLeader", "Spanish_ReservedYoungMan",
            "Spanish_ConfidentWoman", "Spanish_ThoughtfulMan", "Spanish_Strong-WilledBoy",
            "Spanish_SophisticatedLady", "Spanish_RationalMan", "Spanish_AnimeCharacter",
            "Spanish_ElegantNarrator", "Spanish_GentlemanNarrator", "Spanish_YoungBoy",
            "Spanish_SweetGirl", "Spanish_SeniorLady", "Spanish_CalmVoice",
            "Spanish_FreshVoice", "Spanish_EnergeticVoice", "Spanish_SeriousTone",
            "Spanish_WarmVoice", "Spanish_BrightVoice", "Spanish_MysteriousVoice",
            "Spanish_AncientVoice", "Spanish_DynamicVoice", "Spanish_SoftVoice",
            "Spanish_GentleVoice", "Spanish_LivelyVoice", "Spanish_SmoothVoice",
            "Spanish_FirmVoice", "Spanish_FriendlyVoice", "Spanish_SophisticatedVoice"
        ]
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
                "model": (["viduq2-pro", "viduq2-turbo", "viduq1", "viduq1-classic", "vidu2.0", "vidu1.5"], 
                         {"default": "viduq2-pro"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "audio": ("BOOLEAN", {"default": False}),
                "voice_language": (["中文(普通话)", "中文(粤语)", "English", "日本語", "한국어", "Español"], 
                                 {"default": "中文(普通话)"}),
                "voice_id": ([""] + cls.VOICE_OPTIONS["中文(普通话)"], {"default": ""}),
                "is_rec": ("BOOLEAN", {"default": False}),
                "duration": ("INT", {"default": 5, "min": 1, "max": 8}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "resolution": (["360p", "540p", "720p", "1080p"], {"default": "720p"}),
                "movement_amplitude": (["auto", "small", "medium", "large"], {"default": "auto"}),
                "bgm": ("BOOLEAN", {"default": False}),
                "off_peak": ("BOOLEAN", {"default": False}),
                "watermark": ("BOOLEAN", {"default": False}),
                "wm_position": ([1, 2, 3, 4], {"default": 3}),
            }
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "task_id", "response")
    FUNCTION = "generate_video"
    CATEGORY = CATEGORY_TYPE

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 900

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def image_to_base64(self, image_tensor):
        """Convert tensor to base64 string with data URI prefix"""
        if image_tensor is None:
            return None

        pil_image = tensor2pil_naku(image_tensor)[0]

        # Resize image if too large to prevent "request entity too large" error
        # Maintain aspect ratio by resizing the longest side to 1920 pixels
        max_dimension = 1920
        original_width, original_height = pil_image.size

        if original_width > max_dimension or original_height > max_dimension:
            # Calculate the scaling factor to maintain aspect ratio
            scale_factor = max_dimension / max(original_width, original_height)
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)

            # Resize the image using LANCZOS resampling for better quality
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return base64_str  # Return pure base64 string without data URI prefix

    def generate_video(self, image, prompt, model="viduq2-pro", api_key="",
                      audio=False, voice_language="中文(普通话)", voice_id="male-qn-jingying", 
                      is_rec=False, duration=5, seed=0, resolution="720p", 
                      movement_amplitude="auto", bgm=False, off_peak=False, 
                      watermark=False, wm_position=3):
        
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key
            
        if not self.api_key:
            error_response = {"status": "error", "message": "API key not provided or not found in config"}
            return ("", "", "", json.dumps(error_response))
            
        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            image_base64 = self.image_to_base64(image)
            if not image_base64:
                error_message = "Failed to convert image to base64"
                print(error_message)
                return ("", "", "", json.dumps({"status": "error", "message": error_message}))

            payload = {
                "model": model,
                "images": [image_base64],  
                "duration": duration,
                "seed": seed if seed > 0 else 0,
                "resolution": resolution,
                "movement_amplitude": movement_amplitude,
                "off_peak": off_peak
            }

            if prompt.strip():
                payload["prompt"] = prompt

            if audio:
                payload["audio"] = True
                payload["voice_id"] = voice_id
            
            if is_rec:
                payload["is_rec"] = True
            
            if bgm:
                payload["bgm"] = True
            
            if watermark:
                payload["watermark"] = True
                payload["wm_position"] = wm_position

            pbar.update_absolute(20)

            response = requests.post(
                f"{get_baseurl()}/vidu/v2/img2video",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            pbar.update_absolute(30)
            
            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(error_message)
                return ("", "", "", json.dumps({"status": "error", "message": error_message}))
                
            result = response.json()
            
            if "task_id" not in result:
                error_message = f"No task_id in response: {result}"
                print(error_message)
                return ("", "", "", json.dumps({"status": "error", "message": error_message}))
                
            task_id = result.get("task_id")
            
            pbar.update_absolute(40)

            max_attempts = 120
            attempts = 0
            video_url = None
            
            while attempts < max_attempts:
                time.sleep(10)
                attempts += 1
                
                try:
                    status_response = requests.get(
                        f"{get_baseurl()}/vidu/v2/tasks/{task_id}/creations",
                        headers=self.get_headers(),
                        timeout=self.timeout
                    )
                    
                    if status_response.status_code != 200:
                        print(f"Status check failed: {status_response.status_code} - {status_response.text}")
                        continue
                        
                    status_result = status_response.json()
                    
                    state = status_result.get("state", "")

                    progress_value = min(90, 40 + (attempts * 50 // max_attempts))
                    pbar.update_absolute(progress_value)
                    
                    if state == "success":
                        creations = status_result.get("creations", [])
                        if creations and len(creations) > 0:
                            video_url = creations[0].get("url", "")
                            if video_url:
                                print(f"Video URL found: {video_url}")
                                break
                    elif state == "failed":
                        err_code = status_result.get("err_code", "Unknown error")
                        error_message = f"Video generation failed: {err_code}"
                        print(error_message)
                        return ("", "", task_id, json.dumps({"status": "error", "message": error_message}))
                        
                except Exception as e:
                    print(f"Error checking generation status (attempt {attempts}): {str(e)}")
            
            if not video_url:
                error_message = f"Failed to retrieve video URL after {max_attempts} attempts"
                print(error_message)
                return ("", "", task_id, json.dumps({"status": "error", "message": error_message}))
            
            pbar.update_absolute(95)
            print(f"Video generation completed. URL: {video_url}")

            video_adapter = ComflyVideoAdapter(video_url)
            
            response_data = {
                "status": "success",
                "task_id": task_id,
                "video_url": video_url,
                "model": model,
                "duration": duration,
                "resolution": resolution,
                "seed": result.get("seed", seed),
                "voice_language": voice_language if audio else "N/A",
                "voice_id": voice_id if audio else "N/A"
            }

            pbar.update_absolute(100)
            video_adapter = ComflyVideoAdapter(video_url)
            return (video_adapter, video_url, task_id, json.dumps(response_data))
            
        except Exception as e:
            error_message = f"Error generating video: {str(e)}"
            print(error_message)
            import traceback
            traceback.print_exc()
            return ("", "", "", json.dumps({"status": "error", "message": error_message}))


class NakuNodeAPI_vidu_text2video:
    """
    NakuNodeAPI Vidu Text to Video node
    Generates videos from text prompts using Vidu API
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (["viduq2", "viduq1", "vidu1.5"], {"default": "viduq2"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "style": (["general", "anime"], {"default": "general"}),
                "duration": ("INT", {"default": 5, "min": 1, "max": 10}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "aspect_ratio": (["16:9", "9:16", "3:4", "4:3", "1:1"], {"default": "16:9"}),
                "resolution": (["360p", "540p", "720p", "1080p"], {"default": "720p"}),
                "movement_amplitude": (["auto", "small", "medium", "large"], {"default": "auto"}),
                "bgm": ("BOOLEAN", {"default": False}),
                "off_peak": ("BOOLEAN", {"default": False}),
                "watermark": ("BOOLEAN", {"default": False}),
                "wm_position": ([1, 2, 3, 4], {"default": 3}),
            }
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "task_id", "response")
    FUNCTION = "generate_video"
    CATEGORY = CATEGORY_TYPE

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 900

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def generate_video(self, prompt, model="viduq2", api_key="", style="general",
                      duration=5, seed=0, aspect_ratio="16:9", resolution="720p",
                      movement_amplitude="auto", bgm=False, off_peak=False,
                      watermark=False, wm_position=3):
        
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key
            
        if not self.api_key:
            error_response = {"status": "error", "message": "API key not provided or not found in config"}
            return ("", "", "", json.dumps(error_response))
            
        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            payload = {
                "model": model,
                "prompt": prompt,
                "duration": duration,
                "seed": seed if seed > 0 else 0,
                "aspect_ratio": aspect_ratio,
                "resolution": resolution,
                "movement_amplitude": movement_amplitude,
                "off_peak": off_peak
            }

            if model != "viduq2":
                payload["style"] = style
            
            if bgm:
                payload["bgm"] = True
            
            if watermark:
                payload["watermark"] = True
                payload["wm_position"] = wm_position

            pbar.update_absolute(20)

            response = requests.post(
                f"{get_baseurl()}/vidu/v2/text2video",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            pbar.update_absolute(30)
            
            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(error_message)
                return ("", "", "", json.dumps({"status": "error", "message": error_message}))
                
            result = response.json()
            
            if "task_id" not in result:
                error_message = f"No task_id in response: {result}"
                print(error_message)
                return ("", "", "", json.dumps({"status": "error", "message": error_message}))
                
            task_id = result.get("task_id")
            
            pbar.update_absolute(40)

            max_attempts = 120
            attempts = 0
            video_url = None
            
            while attempts < max_attempts:
                time.sleep(10)
                attempts += 1
                
                try:
                    status_response = requests.get(
                        f"{get_baseurl()}/vidu/v2/tasks/{task_id}/creations",
                        headers=self.get_headers(),
                        timeout=self.timeout
                    )
                    
                    if status_response.status_code != 200:
                        print(f"Status check failed: {status_response.status_code} - {status_response.text}")
                        continue
                        
                    status_result = status_response.json()
                    
                    state = status_result.get("state", "")

                    progress_value = min(90, 40 + (attempts * 50 // max_attempts))
                    pbar.update_absolute(progress_value)
                    
                    if state == "success":
                        creations = status_result.get("creations", [])
                        if creations and len(creations) > 0:
                            video_url = creations[0].get("url", "")
                            if video_url:
                                print(f"Video URL found: {video_url}")
                                break
                    elif state == "failed":
                        err_code = status_result.get("err_code", "Unknown error")
                        error_message = f"Video generation failed: {err_code}"
                        print(error_message)
                        return ("", "", task_id, json.dumps({"status": "error", "message": error_message}))
                        
                except Exception as e:
                    print(f"Error checking generation status (attempt {attempts}): {str(e)}")
            
            if not video_url:
                error_message = f"Failed to retrieve video URL after {max_attempts} attempts"
                print(error_message)
                return ("", "", task_id, json.dumps({"status": "error", "message": error_message}))
            
            pbar.update_absolute(95)
            print(f"Video generation completed. URL: {video_url}")

            video_adapter = ComflyVideoAdapter(video_url)
            
            response_data = {
                "status": "success",
                "task_id": task_id,
                "video_url": video_url,
                "model": model,
                "duration": duration,
                "resolution": resolution,
                "aspect_ratio": aspect_ratio,
                "seed": result.get("seed", seed)
            }

            pbar.update_absolute(100)
            video_adapter = ComflyVideoAdapter(video_url)
            return (video_adapter, video_url, task_id, json.dumps(response_data))
            
        except Exception as e:
            error_message = f"Error generating video: {str(e)}"
            print(error_message)
            import traceback
            traceback.print_exc()
            return ("", "", "", json.dumps({"status": "error", "message": error_message}))


class NakuNodeAPI_vidu_ref2video:
    """
    NakuNodeAPI Vidu Reference to Video node
    Generates videos from reference images with optional audio
    """
    
    VOICE_OPTIONS = NakuNodeAPI_vidu_img2video.VOICE_OPTIONS

    @classmethod
    def INPUT_TYPES(cls):
        all_voices = [""]
        for lang, voices in cls.VOICE_OPTIONS.items():
            all_voices.extend(voices)

        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (["viduq2", "viduq1", "vidu2.0", "vidu1.5"],
                         {"default": "viduq2"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "image7": ("IMAGE",),
                "audio": ("BOOLEAN", {"default": False}),
                "subject1_id": ("STRING", {"default": "1"}),
                "subject1_voice_id": (all_voices, {"default": ""}),
                "subject2_id": ("STRING", {"default": "2"}),
                "subject2_voice_id": (all_voices, {"default": ""}),
                "subject3_id": ("STRING", {"default": "3"}),
                "subject3_voice_id": (all_voices, {"default": ""}),
                "duration": ("INT", {"default": 5, "min": 1, "max": 10}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "aspect_ratio": (["16:9", "9:16", "4:3", "3:4", "1:1"], {"default": "16:9"}),
                "resolution": (["540p", "720p", "1080p"], {"default": "720p"}),
                "movement_amplitude": (["auto", "small", "medium", "large"], {"default": "auto"}),
                "bgm": ("BOOLEAN", {"default": False}),
                "off_peak": ("BOOLEAN", {"default": False}),
                "watermark": ("BOOLEAN", {"default": False}),
                "wm_position": ([1, 2, 3, 4], {"default": 3}),
            }
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "task_id", "response")
    FUNCTION = "generate_video"
    CATEGORY = CATEGORY_TYPE

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 900

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def image_to_base64(self, image_tensor):
        """Convert tensor to base64 string with data URI prefix"""
        if image_tensor is None:
            return None

        pil_image = tensor2pil_naku(image_tensor)[0]

        # Resize image if too large to prevent "request entity too large" error
        # Maintain aspect ratio by resizing the longest side to 1920 pixels
        max_dimension = 1920
        original_width, original_height = pil_image.size

        if original_width > max_dimension or original_height > max_dimension:
            # Calculate the scaling factor to maintain aspect ratio
            scale_factor = max_dimension / max(original_width, original_height)
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)

            # Resize the image using LANCZOS resampling for better quality
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return base64_str  # Return pure base64 string without data URI prefix
    
    def generate_video(self, prompt, model="viduq2", api_key="",
                      image1=None, image2=None, image3=None, image4=None,
                      image5=None, image6=None, image7=None,
                      audio=False, subject1_id="1", subject1_voice_id="",
                      subject2_id="2", subject2_voice_id="",
                      subject3_id="3", subject3_voice_id="",
                      duration=5, seed=0, aspect_ratio="16:9", resolution="720p",
                      movement_amplitude="auto", bgm=False, off_peak=False,
                      watermark=False, wm_position=3):
        
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key
            
        if not self.api_key:
            error_response = {"status": "error", "message": "API key not provided or not found in config"}
            return ("", "", "", json.dumps(error_response))
            
        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            all_images = [image1, image2, image3, image4, image5, image6, image7]
            image_base64_list = []
            
            for img in all_images:
                if img is not None:
                    img_base64 = self.image_to_base64(img)
                    if img_base64:
                        image_base64_list.append(img_base64)
            
            if not image_base64_list:
                error_message = "No images provided. At least one image is required."
                print(error_message)
                return ("", "", "", json.dumps({"status": "error", "message": error_message}))

            payload = {
                "model": model,
                "prompt": prompt,
                "duration": duration,
                "seed": seed if seed > 0 else 0,
                "aspect_ratio": aspect_ratio,
                "resolution": resolution,
                "movement_amplitude": movement_amplitude,
                "off_peak": off_peak
            }

            if audio:
                subjects = []
 
                subject_images = [[], [], []]
                for i, img_b64 in enumerate(image_base64_list):
                    subject_idx = min(i // 3, 2)  
                    if len(subject_images[subject_idx]) < 3:
                        subject_images[subject_idx].append(img_b64)

                subject_configs = [
                    (subject1_id, subject1_voice_id, subject_images[0]),
                    (subject2_id, subject2_voice_id, subject_images[1]),
                    (subject3_id, subject3_voice_id, subject_images[2])
                ]
                
                for subj_id, voice_id, images in subject_configs:
                    if images:
                        subject = {
                            "id": subj_id,
                            "images": images,
                            "voice_id": voice_id if voice_id else ""
                        }
                        subjects.append(subject)
                
                if subjects:
                    payload["subjects"] = subjects
                    payload["audio"] = True
            else:
                payload["images"] = image_base64_list
                if bgm:
                    payload["bgm"] = True
            
            if watermark:
                payload["watermark"] = True
                payload["wm_position"] = wm_position

            pbar.update_absolute(20)

            response = requests.post(
                f"{get_baseurl()}/vidu/v2/reference2video",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            pbar.update_absolute(30)
            
            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(error_message)
                return ("", "", "", json.dumps({"status": "error", "message": error_message}))
                
            result = response.json()
            
            if "task_id" not in result:
                error_message = f"No task_id in response: {result}"
                print(error_message)
                return ("", "", "", json.dumps({"status": "error", "message": error_message}))
                
            task_id = result.get("task_id")
            
            pbar.update_absolute(40)

            max_attempts = 120
            attempts = 0
            video_url = None
            
            while attempts < max_attempts:
                time.sleep(10)
                attempts += 1
                
                try:
                    status_response = requests.get(
                        f"{get_baseurl()}/vidu/v2/tasks/{task_id}/creations",
                        headers=self.get_headers(),
                        timeout=self.timeout
                    )
                    
                    if status_response.status_code != 200:
                        print(f"Status check failed: {status_response.status_code} - {status_response.text}")
                        continue
                        
                    status_result = status_response.json()
                    
                    state = status_result.get("state", "")

                    progress_value = min(90, 40 + (attempts * 50 // max_attempts))
                    pbar.update_absolute(progress_value)
                    
                    if state == "success":
                        creations = status_result.get("creations", [])
                        if creations and len(creations) > 0:
                            video_url = creations[0].get("url", "")
                            if video_url:
                                print(f"Video URL found: {video_url}")
                                break
                    elif state == "failed":
                        err_code = status_result.get("err_code", "Unknown error")
                        error_message = f"Video generation failed: {err_code}"
                        print(error_message)
                        return ("", "", task_id, json.dumps({"status": "error", "message": error_message}))
                        
                except Exception as e:
                    print(f"Error checking generation status (attempt {attempts}): {str(e)}")
            
            if not video_url:
                error_message = f"Failed to retrieve video URL after {max_attempts} attempts"
                print(error_message)
                return ("", "", task_id, json.dumps({"status": "error", "message": error_message}))
            
            pbar.update_absolute(95)
            print(f"Video generation completed. URL: {video_url}")

            video_adapter = ComflyVideoAdapter(video_url)
            
            response_data = {
                "status": "success",
                "task_id": task_id,
                "video_url": video_url,
                "model": model,
                "duration": duration,
                "resolution": resolution,
                "aspect_ratio": aspect_ratio,
                "audio": audio,
                "images_count": len(image_base64_list),
                "seed": result.get("seed", seed)
            }

            pbar.update_absolute(100)
            video_adapter = ComflyVideoAdapter(video_url)
            return (video_adapter, video_url, task_id, json.dumps(response_data))
            
        except Exception as e:
            error_message = f"Error generating video: {str(e)}"
            print(error_message)
            import traceback
            traceback.print_exc()
            return ("", "", "", json.dumps({"status": "error", "message": error_message}))


class NakuNodeAPI_kling_o1_video:
    """
    NakuNodeAPI Kling O1-Video node
    Generates videos using Kling O1-Video API
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "model_name": (["kling-video-o1"], {"default": "kling-video-o1"}),
                "mode": (["pro"], {"default": "pro"}),
                "aspect_ratio": (["16:9", "9:16", "1:1"], {"default": "16:9"}),
                "duration": (["3", "4", "5", "6", "7", "8", "9", "10"], {"default": "5"}),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "image7": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "response")
    FUNCTION = "generate_video"
    CATEGORY = CATEGORY_TYPE

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 600

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def image_to_base64(self, image_tensor):
        """Convert tensor to base64 string with data URI prefix"""
        if image_tensor is None:
            return None

        pil_image = tensor2pil_naku(image_tensor)[0]

        # Resize image if too large to prevent "request entity too large" error
        # Maintain aspect ratio by resizing the longest side to 1920 pixels
        max_dimension = 1920
        original_width, original_height = pil_image.size

        if original_width > max_dimension or original_height > max_dimension:
            # Calculate the scaling factor to maintain aspect ratio
            scale_factor = max_dimension / max(original_width, original_height)
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)

            # Resize the image using LANCZOS resampling for better quality
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return base64_str  # Return pure base64 string without data URI prefix

    def generate_video(self, prompt, api_key="", model_name="kling-video-o1", mode="pro",
                      aspect_ratio="", duration="5", image1=None, image2=None, image3=None,
                      image4=None, image5=None, image6=None, image7=None):

        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key

        if not self.api_key:
            error_response = {"status": "error", "message": "API key not provided or not found in config"}
            return ("", "", json.dumps(error_response))

        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            # Convert image tensors to base64 if provided
            image_base64_list = []
            for img in [image1, image2, image3, image4, image5, image6, image7]:
                if img is not None:
                    img_base64 = self.image_to_base64(img)
                    if img_base64:
                        image_base64_list.append(img_base64)

            payload = {
                "model_name": model_name,
                "prompt": prompt,
                "mode": mode,
                "duration": int(duration)
            }

            # Add optional parameters if provided
            if aspect_ratio:
                payload["aspect_ratio"] = aspect_ratio

            # Add images if provided
            if image_base64_list:
                payload["image_list"] = [{"image_url": img_url} for img_url in image_base64_list]

            pbar.update_absolute(30)

            # Make API request
            response = requests.post(
                f"{get_baseurl()}/kling/v1/videos/omni-video",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )

            pbar.update_absolute(40)

            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(error_message)
                return ("", "", json.dumps({"status": "error", "message": error_message}))

            result = response.json()

            # Check for success code
            if result.get("code") != 0:
                error_message = f"API Error: {result.get('message', 'Unknown error')}"
                print(error_message)
                return ("", "", json.dumps({"status": "error", "message": error_message}))

            # Extract task_id from nested data structure
            task_data = result.get("data", {})
            task_id = task_data.get("task_id")

            if not task_id:
                error_message = f"No task_id in response data: {result}"
                print(error_message)
                return ("", "", json.dumps({"status": "error", "message": error_message}))
            pbar.update_absolute(50)

            # Poll for completion
            max_attempts = 180  # 30 minutes with 10-second intervals
            attempts = 0
            video_url = None

            while attempts < max_attempts:
                time.sleep(10)
                attempts += 1

                try:
                    status_response = requests.get(
                        f"{get_baseurl()}/kling/v1/videos/omni-video/{task_id}",
                        headers=self.get_headers(),
                        timeout=self.timeout
                    )

                    if status_response.status_code != 200:
                        print(f"Status check failed: {status_response.status_code} - {status_response.text}")
                        continue

                    status_result = status_response.json()

                    # Check if the API call was successful
                    if status_result.get("code") != 0:
                        error_msg = status_result.get("message", "Unknown error")
                        error_message = f"Status check API error: {error_msg}"
                        print(error_message)
                        return ("", "", json.dumps({"status": "error", "message": error_message}))

                    # Extract status from nested data structure (based on actual API response)
                    task_data = status_result.get("data", {})
                    task_status = task_data.get("task_status", "")
                    progress = task_data.get("progress", "0%")

                    progress_value = min(90, 50 + (attempts * 40 // max_attempts))
                    pbar.update_absolute(progress_value)

                    if task_status == "success" or task_status == "succeed":
                        # Extract video URL from response (based on actual API response structure)
                        task_result = task_data.get("task_result", {})
                        videos = task_result.get("videos", [])
                        if videos and len(videos) > 0:
                            video_url = videos[0].get("url", "")
                            if video_url:
                                print(f"Video URL found: {video_url}")
                                break
                    elif task_status == "failed":
                        error_msg = task_data.get("error_message", task_data.get("task_status_msg", "Unknown error"))
                        error_message = f"Video generation failed: {error_msg}"
                        print(error_message)
                        return ("", "", json.dumps({"status": "error", "message": error_message}))
                    elif task_status == "processing" or task_status == "submitted":
                        continue
                    else:
                        print(f"Unexpected status: {task_status}")

                except Exception as e:
                    print(f"Error checking generation status (attempt {attempts}): {str(e)}")
                    continue

            if not video_url:
                error_message = f"Failed to retrieve video URL after {max_attempts} attempts"
                print(error_message)
                return ("", "", json.dumps({"status": "error", "message": error_message}))

            pbar.update_absolute(100)

            response_data = {
                "status": "success",
                "task_id": task_id,
                "video_url": video_url,
                "model": model_name,
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "duration": duration,
                "mode": mode
            }

            video_adapter = ComflyVideoAdapter(video_url)
            return (video_adapter, video_url, json.dumps(response_data))

        except Exception as e:
            error_message = f"Error generating video: {str(e)}"
            print(error_message)
            import traceback
            traceback.print_exc()
            empty_video = ComflyVideoAdapter("")
            return (empty_video, "", json.dumps({"status": "error", "message": error_message}))


class NakuNodeAPI_vidu_start_end2video:
    """
    NakuNodeAPI Vidu Start-End Frame to Video node
    Generates videos from start and end frame images
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start_image": ("IMAGE",),
                "end_image": ("IMAGE",),
                "model": (["viduq2-pro", "viduq2-turbo", "viduq1", "viduq1-classic", "vidu2.0", "vidu1.5"], 
                         {"default": "viduq2-pro"}),
            },
            "optional": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "api_key": ("STRING", {"default": ""}),
                "is_rec": ("BOOLEAN", {"default": False}),
                "duration": ("INT", {"default": 5, "min": 1, "max": 8}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "resolution": (["360p", "540p", "720p", "1080p"], {"default": "720p"}),
                "movement_amplitude": (["auto", "small", "medium", "large"], {"default": "auto"}),
                "bgm": ("BOOLEAN", {"default": False}),
                "off_peak": ("BOOLEAN", {"default": False}),
                "watermark": ("BOOLEAN", {"default": False}),
                "wm_position": ([1, 2, 3, 4], {"default": 3}),
            }
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "task_id", "response")
    FUNCTION = "generate_video"
    CATEGORY = CATEGORY_TYPE

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 900

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def image_to_base64(self, image_tensor):
        """Convert tensor to base64 string with data URI prefix"""
        if image_tensor is None:
            return None

        pil_image = tensor2pil_naku(image_tensor)[0]

        # Resize image if too large to prevent "request entity too large" error
        # Maintain aspect ratio by resizing the longest side to 1920 pixels
        max_dimension = 1920
        original_width, original_height = pil_image.size

        if original_width > max_dimension or original_height > max_dimension:
            # Calculate the scaling factor to maintain aspect ratio
            scale_factor = max_dimension / max(original_width, original_height)
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)

            # Resize the image using LANCZOS resampling for better quality
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return base64_str  # Return pure base64 string without data URI prefix
    
    def generate_video(self, start_image, end_image, model="viduq2-pro", prompt="", api_key="", 
                      is_rec=False, duration=5, seed=0, resolution="720p", 
                      movement_amplitude="auto", bgm=False, off_peak=False, 
                      watermark=False, wm_position=3):
        
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key
            
        if not self.api_key:
            error_response = {"status": "error", "message": "API key not provided or not found in config"}
            return ("", "", "", json.dumps(error_response))
            
        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            start_base64 = self.image_to_base64(start_image)
            end_base64 = self.image_to_base64(end_image)
            
            if not start_base64 or not end_base64:
                error_message = "Failed to convert start or end image to base64"
                print(error_message)
                return ("", "", "", json.dumps({"status": "error", "message": error_message}))

            payload = {
                "model": model,
                "start_image": start_base64,
                "end_image": end_base64,
                "duration": duration,
                "seed": seed if seed > 0 else 0,
                "resolution": resolution,
                "movement_amplitude": movement_amplitude,
                "off_peak": off_peak
            }

            if prompt.strip():
                payload["prompt"] = prompt

            if is_rec:
                payload["is_rec"] = True
            
            if bgm:
                payload["bgm"] = True
            
            if watermark:
                payload["watermark"] = True
                payload["wm_position"] = wm_position

            pbar.update_absolute(20)

            response = requests.post(
                f"{get_baseurl()}/vidu/v2/start-end2video",
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            pbar.update_absolute(30)
            
            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(error_message)
                return ("", "", "", json.dumps({"status": "error", "message": error_message}))
                
            result = response.json()
            
            if "task_id" not in result:
                error_message = f"No task_id in response: {result}"
                print(error_message)
                return ("", "", "", json.dumps({"status": "error", "message": error_message}))
                
            task_id = result.get("task_id")
            
            pbar.update_absolute(40)

            max_attempts = 120
            attempts = 0
            video_url = None
            
            while attempts < max_attempts:
                time.sleep(10)
                attempts += 1
                
                try:
                    status_response = requests.get(
                        f"{get_baseurl()}/vidu/v2/tasks/{task_id}/creations",
                        headers=self.get_headers(),
                        timeout=self.timeout
                    )
                    
                    if status_response.status_code != 200:
                        print(f"Status check failed: {status_response.status_code} - {status_response.text}")
                        continue
                        
                    status_result = status_response.json()
                    
                    state = status_result.get("state", "")

                    progress_value = min(90, 40 + (attempts * 50 // max_attempts))
                    pbar.update_absolute(progress_value)
                    
                    if state == "success":
                        creations = status_result.get("creations", [])
                        if creations and len(creations) > 0:
                            video_url = creations[0].get("url", "")
                            if video_url:
                                print(f"Video URL found: {video_url}")
                                break
                    elif state == "failed":
                        err_code = status_result.get("err_code", "Unknown error")
                        error_message = f"Video generation failed: {err_code}"
                        print(error_message)
                        return ("", "", task_id, json.dumps({"status": "error", "message": error_message}))
                        
                except Exception as e:
                    print(f"Error checking generation status (attempt {attempts}): {str(e)}")
            
            if not video_url:
                error_message = f"Failed to retrieve video URL after {max_attempts} attempts"
                print(error_message)
                return ("", "", task_id, json.dumps({"status": "error", "message": error_message}))
            
            pbar.update_absolute(95)
            print(f"Video generation completed. URL: {video_url}")

            video_adapter = ComflyVideoAdapter(video_url)
            
            response_data = {
                "status": "success",
                "task_id": task_id,
                "video_url": video_url,
                "model": model,
                "duration": duration,
                "resolution": resolution,
                "seed": result.get("seed", seed)
            }

            pbar.update_absolute(100)
            video_adapter = ComflyVideoAdapter(video_url)
            return (video_adapter, video_url, task_id, json.dumps(response_data))
            
        except Exception as e:
            error_message = f"Error generating video: {str(e)}"
            print(error_message)
            import traceback
            traceback.print_exc()
            return ("", "", "", json.dumps({"status": "error", "message": error_message}))


# NODE MAPPINGS
NODE_CLASS_MAPPINGS = {
    # Google Nodes
    "NakuNodeAPI_Googel_Veo3": NakuNodeAPI_Googel_Veo3,
    "NakuNodeAPI_nano_banana": NakuNodeAPI_nano_banana,
    "NakuNodeAPI_nano_banana_edit": NakuNodeAPI_nano_banana_edit,
    "NakuNodeAPI_nano_banana2_edit": NakuNodeAPI_nano_banana2_edit,
    "NakuNodeAPI_GeminiTextOnly": NakuNodeAPI_GeminiTextOnly,

    # Kling Nodes
    "NakuNodeAPI_kling_text2video": NakuNodeAPI_kling_text2video,
    "NakuNodeAPI_kling_image2video": NakuNodeAPI_kling_image2video,
    "NakuNodeAPI_kling_multi_image2video": NakuNodeAPI_kling_multi_image2video,
    "NakuNodeAPI_video_extend": NakuNodeAPI_video_extend,
    "NakuNodeAPI_lip_sync": NakuNodeAPI_lip_sync,
    "NakuNodeAPI_kling_o1_video": NakuNodeAPI_kling_o1_video,

    # Midjourney Nodes
    "NakuNodeAPI_upload": NakuNodeAPI_upload,
    "NakuNodeAPI_Mj": NakuNodeAPI_Mj,
    "NakuNodeAPI_Mju": NakuNodeAPI_Mju,
    "NakuNodeAPI_Mjv": NakuNodeAPI_Mjv,
    "NakuNodeAPI_Mj_swap_face": NakuNodeAPI_Mj_swap_face,
    "NakuNodeAPI_mj_video": NakuNodeAPI_mj_video,
    "NakuNodeAPI_mj_video_extend": NakuNodeAPI_mj_video_extend,

    # OpenAI Nodes
    "NakuNodeAPI_gpt_image_edit": NakuNodeAPI_gpt_image_edit,
    "NakuNodeAPI_gpt_image": NakuNodeAPI_gpt_image,
    "NakuNodeAPI_ChatGPTApi": NakuNodeAPI_ChatGPTApi,
    "NakuNodeAPI_sora2": NakuNodeAPI_sora2,
    "NakuNodeAPI_sora2_character": NakuNodeAPI_sora2_character,

    # Vidu Nodes
    "NakuNodeAPI_vidu_img2video": NakuNodeAPI_vidu_img2video,
    "NakuNodeAPI_vidu_text2video": NakuNodeAPI_vidu_text2video,
    "NakuNodeAPI_vidu_ref2video": NakuNodeAPI_vidu_ref2video,
    "NakuNodeAPI_vidu_start_end2video": NakuNodeAPI_vidu_start_end2video,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Google Nodes
    "NakuNodeAPI_Googel_Veo3": "NakuNodeAPI Google Veo3",
    "NakuNodeAPI_nano_banana": "NakuNodeAPI nano_banana",
    "NakuNodeAPI_nano_banana_edit": "NakuNodeAPI nano_banana_edit",
    "NakuNodeAPI_nano_banana2_edit": "NakuNodeAPI nano_banana2_edit",
    "NakuNodeAPI_GeminiTextOnly": "NakuNodeAPI Gemini Text Only",

    # Kling Nodes
    "NakuNodeAPI_kling_text2video": "NakuNodeAPI Kling Text2Video",
    "NakuNodeAPI_kling_image2video": "NakuNodeAPI Kling Image2Video",
    "NakuNodeAPI_kling_multi_image2video": "NakuNodeAPI Kling Multi Image2Video",
    "NakuNodeAPI_video_extend": "NakuNodeAPI Video Extend",
    "NakuNodeAPI_lip_sync": "NakuNodeAPI Lip Sync",
    "NakuNodeAPI_kling_o1_video": "NakuNodeAPI Kling O1-Video",

    # Midjourney Nodes
    "NakuNodeAPI_upload": "NakuNodeAPI MJ Upload",
    "NakuNodeAPI_Mj": "NakuNodeAPI MJ",
    "NakuNodeAPI_Mju": "NakuNodeAPI MJU",
    "NakuNodeAPI_Mjv": "NakuNodeAPI MJV",
    "NakuNodeAPI_Mj_swap_face": "NakuNodeAPI MJ Face Swap",
    "NakuNodeAPI_mj_video": "NakuNodeAPI MJ Video",
    "NakuNodeAPI_mj_video_extend": "NakuNodeAPI MJ Video Extend",

    # OpenAI Nodes
    "NakuNodeAPI_gpt_image_edit": "NakuNodeAPI GPT Image Edit",
    "NakuNodeAPI_gpt_image": "NakuNodeAPI GPT Image",
    "NakuNodeAPI_ChatGPTApi": "NakuNodeAPI ChatGPT API",
    "NakuNodeAPI_sora2": "NakuNodeAPI Sora2",
    "NakuNodeAPI_sora2_character": "NakuNodeAPI Sora2 Character",

    # Vidu Nodes
    "NakuNodeAPI_vidu_img2video": "NakuNodeAPI Vidu Image2Video",
    "NakuNodeAPI_vidu_text2video": "NakuNodeAPI Vidu Text2Video",
    "NakuNodeAPI_vidu_ref2video": "NakuNodeAPI Vidu Ref2Video",
    "NakuNodeAPI_vidu_start_end2video": "NakuNodeAPI Vidu Start-End2Video",
}

# Global variable for baseurl that will be shared
global_naku_baseurl = "https://ai.comfly.chat"

def get_naku_baseurl():
    """获取当前的 API 基础地址"""
    global global_naku_baseurl
    return global_naku_baseurl

def get_naku_config():
    """获取配置"""
    try:
        import os
        config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Comflyapi.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        else:
            # Try looking in parent directory
            parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            config_path = os.path.join(parent_dir, 'Comflyapi.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                return config
            else:
                return {}
    except:
        return {}

def save_naku_config(config):
    """保存配置"""
    import os
    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Comflyapi.json')
    if os.path.exists(config_path):
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
    else:
        # Try saving to the parent directory
        parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        config_path = os.path.join(parent_dir, 'Comflyapi.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)

class NakuNodeAPI_api_set:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_base": (["comfly", "ip", "hk", "us"], {"default": "comfly"}),
                "apikey": ("STRING", {"default": ""}),
            },
            "optional": {
                "custom_ip": ("STRING", {"default": "", "placeholder": "Enter IP when using 'ip' option"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("apikey",)
    FUNCTION = "set_api_base"
    CATEGORY = "NakuNodes/API"

    def set_api_base(self, api_base, apikey="", custom_ip=""):
        global global_naku_baseurl

        base_url_mapping = {
            "comfly": "https://ai.comfly.chat",
            "ip": custom_ip,
            "hk": "https://hk-api.gptbest.vip",
            "us": "https://api.gptbest.vip"
        }

        if api_base == "ip" and not custom_ip.strip():
            raise ValueError("When selecting 'ip' option, you must provide a custom IP address in the 'custom_ip' field")

        if api_base in base_url_mapping:
            selected_url = base_url_mapping[api_base]
            # Validate that when ip is selected, custom_ip is provided
            if api_base == "ip":
                if not custom_ip or not custom_ip.strip():
                    raise ValueError("When selecting 'ip' option, you must provide a custom IP address in the 'custom_ip' field")
                # Ensure the custom IP has proper protocol
                if not custom_ip.startswith(('http://', 'https://')):
                    selected_url = f"https://{custom_ip}"
                else:
                    selected_url = custom_ip
            global global_naku_baseurl
            global_naku_baseurl = selected_url

        if apikey.strip():
            config = get_naku_config()
            config['api_key'] = apikey
            save_naku_config(config)

        message = f"API Base URL set to: {global_naku_baseurl}"
        if apikey.strip():
            message += "\nAPI key has been updated"

        print(message)
        return (apikey,)


# Updated NODE_CLASS_MAPPINGS with the api_set node
NODE_CLASS_MAPPINGS = {
    # Google Nodes
    "NakuNodeAPI_Googel_Veo3": NakuNodeAPI_Googel_Veo3,
    "NakuNodeAPI_nano_banana": NakuNodeAPI_nano_banana,
    "NakuNodeAPI_nano_banana_edit": NakuNodeAPI_nano_banana_edit,
    "NakuNodeAPI_nano_banana2_edit": NakuNodeAPI_nano_banana2_edit,
    "NakuNodeAPI_GeminiTextOnly": NakuNodeAPI_GeminiTextOnly,

    # Kling Nodes
    "NakuNodeAPI_kling_text2video": NakuNodeAPI_kling_text2video,
    "NakuNodeAPI_kling_image2video": NakuNodeAPI_kling_image2video,
    "NakuNodeAPI_kling_multi_image2video": NakuNodeAPI_kling_multi_image2video,
    "NakuNodeAPI_video_extend": NakuNodeAPI_video_extend,
    "NakuNodeAPI_lip_sync": NakuNodeAPI_lip_sync,
    "NakuNodeAPI_kling_o1_video": NakuNodeAPI_kling_o1_video,

    # Midjourney Nodes
    "NakuNodeAPI_upload": NakuNodeAPI_upload,
    "NakuNodeAPI_Mj": NakuNodeAPI_Mj,
    "NakuNodeAPI_Mju": NakuNodeAPI_Mju,
    "NakuNodeAPI_Mjv": NakuNodeAPI_Mjv,
    "NakuNodeAPI_Mj_swap_face": NakuNodeAPI_Mj_swap_face,
    "NakuNodeAPI_mj_video": NakuNodeAPI_mj_video,
    "NakuNodeAPI_mj_video_extend": NakuNodeAPI_mj_video_extend,

    # OpenAI Nodes
    "NakuNodeAPI_gpt_image_edit": NakuNodeAPI_gpt_image_edit,
    "NakuNodeAPI_gpt_image": NakuNodeAPI_gpt_image,
    "NakuNodeAPI_ChatGPTApi": NakuNodeAPI_ChatGPTApi,
    "NakuNodeAPI_sora2": NakuNodeAPI_sora2,
    "NakuNodeAPI_sora2_character": NakuNodeAPI_sora2_character,

    # API Set Node
    "NakuNodeAPI_api_set": NakuNodeAPI_api_set,

    # Vidu Nodes
    "NakuNodeAPI_vidu_img2video": NakuNodeAPI_vidu_img2video,
    "NakuNodeAPI_vidu_text2video": NakuNodeAPI_vidu_text2video,
    "NakuNodeAPI_vidu_ref2video": NakuNodeAPI_vidu_ref2video,
    "NakuNodeAPI_vidu_start_end2video": NakuNodeAPI_vidu_start_end2video,
}

# Updated NODE_DISPLAY_NAME_MAPPINGS with the api_set node
NODE_DISPLAY_NAME_MAPPINGS = {
    # Google Nodes
    "NakuNodeAPI_Googel_Veo3": "NakuNodeAPI Google Veo3",
    "NakuNodeAPI_nano_banana": "NakuNodeAPI nano_banana",
    "NakuNodeAPI_nano_banana_edit": "NakuNodeAPI nano_banana_edit",
    "NakuNodeAPI_nano_banana2_edit": "NakuNodeAPI nano_banana2_edit",
    "NakuNodeAPI_GeminiTextOnly": "NakuNodeAPI Gemini Text Only",

    # Kling Nodes
    "NakuNodeAPI_kling_text2video": "NakuNodeAPI Kling Text2Video",
    "NakuNodeAPI_kling_image2video": "NakuNodeAPI Kling Image2Video",
    "NakuNodeAPI_kling_multi_image2video": "NakuNodeAPI Kling Multi Image2Video",
    "NakuNodeAPI_video_extend": "NakuNodeAPI Video Extend",
    "NakuNodeAPI_lip_sync": "NakuNodeAPI Lip Sync",
    "NakuNodeAPI_kling_o1_video": "NakuNodeAPI Kling O1-Video",

    # Midjourney Nodes
    "NakuNodeAPI_upload": "NakuNodeAPI MJ Upload",
    "NakuNodeAPI_Mj": "NakuNodeAPI MJ",
    "NakuNodeAPI_Mju": "NakuNodeAPI MJU",
    "NakuNodeAPI_Mjv": "NakuNodeAPI MJV",
    "NakuNodeAPI_Mj_swap_face": "NakuNodeAPI MJ Face Swap",
    "NakuNodeAPI_mj_video": "NakuNodeAPI MJ Video",
    "NakuNodeAPI_mj_video_extend": "NakuNodeAPI MJ Video Extend",

    # OpenAI Nodes
    "NakuNodeAPI_gpt_image_edit": "NakuNodeAPI GPT Image Edit",
    "NakuNodeAPI_gpt_image": "NakuNodeAPI GPT Image",
    "NakuNodeAPI_ChatGPTApi": "NakuNodeAPI ChatGPT API",
    "NakuNodeAPI_sora2": "NakuNodeAPI Sora2",
    "NakuNodeAPI_sora2_character": "NakuNodeAPI Sora2 Character",

    # API Set Node
    "NakuNodeAPI_api_set": "NakuNodeAPI API Settings",

    # Vidu Nodes
    "NakuNodeAPI_vidu_img2video": "NakuNodeAPI Vidu Image2Video",
    "NakuNodeAPI_vidu_text2video": "NakuNodeAPI Vidu Text2Video",
    "NakuNodeAPI_vidu_ref2video": "NakuNodeAPI Vidu Ref2Video",
    "NakuNodeAPI_vidu_start_end2video": "NakuNodeAPI Vidu Start-End2Video",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]