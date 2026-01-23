import os
import httpx
import base64
import mimetypes
from pathlib import Path
from typing import Optional, List, Any, Dict
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
VOLC_API_KEY = os.getenv("VOLC_API_KEY")
DEFAULT_IMAGE_MODEL = os.getenv("VOLC_IMAGE_MODEL")
DEFAULT_VIDEO_MODEL = os.getenv("VOLC_VIDEO_MODEL")

API_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"

# Initialize MCP Server
mcp = FastMCP("jimeng_visual_generation")

# --- Helper Functions ---

def _process_image_input(image_input: str) -> str:
    """
    Process image input string.
    - If it's a URL or Base64 string, return as is.
    - If it's a local file path, convert to Base64 data URI.
    """
    # 1. Check if it's already a Data URI or URL
    if image_input.startswith("http") or image_input.startswith("data:"):
        return image_input
    
    # 2. Check if it's a local file
    file_path = Path(image_input)
    if file_path.exists() and file_path.is_file():
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            # Fallback for common image types if mime detection fails
            suffix = file_path.suffix.lower()
            if suffix in ['.jpg', '.jpeg']: mime_type = 'image/jpeg'
            elif suffix == '.png': mime_type = 'image/png'
            elif suffix == '.webp': mime_type = 'image/webp'
            else: mime_type = 'application/octet-stream'
            
        with open(file_path, "rb") as f:
            encoded_string = base64.b64encode(f.read()).decode('utf-8')
            return f"data:{mime_type};base64,{encoded_string}"
            
    # 3. Return original if we can't process it (API might handle it or error out)
    return image_input

async def _make_api_request(method: str, endpoint: str, json_data: Optional[Dict] = None, params: Optional[Dict] = None) -> Dict:
    """Helper to make API requests to Volcengine."""
    if not VOLC_API_KEY:
        raise ValueError("VOLC_API_KEY environment variable is not set.")

    url = f"{API_BASE_URL}{endpoint}"
    headers = {
        "Authorization": f"Bearer {VOLC_API_KEY}",
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.request(method, url, headers=headers, json=json_data, params=params)
        if response.status_code >= 400:
            error_msg = f"API Error {response.status_code}: {response.text}"
            try:
                error_data = response.json()
                if "error" in error_data:
                    error_msg = f"API Error {response.status_code}: {error_data['error'].get('message', response.text)}"
            except:
                pass
            raise httpx.HTTPStatusError(error_msg, request=response.request, response=response)
        
        return response.json()

# --- Image Generation ---

class ImageResolution(str, Enum):
    RES_2K = "2K"
    RES_4K = "4K"

class ImageRatio(str, Enum):
    SQUARE = "1:1"
    PORTRAIT_4_3 = "3:4"
    PORTRAIT_16_9 = "9:16"
    LANDSCAPE_4_3 = "4:3"
    LANDSCAPE_16_9 = "16:9"

class GenerateImageInput(BaseModel):
    prompt: str = Field(..., description="Text prompt for image generation. Describes the content, style, and mood of the desired image.")
    model: Optional[str] = Field(default=DEFAULT_IMAGE_MODEL, description="Model ID to use (e.g., 'doubao-seedream-4.5').")
    image_urls: Optional[List[str]] = Field(default=None, description="List of reference images for Image-to-Image (I2I) or Multi-Image fusion. Supports:\n1. HTTP/HTTPS URLs\n2. Base64 Data URIs\n3. Local file paths (e.g., 'C:/images/ref.jpg') - will be automatically converted.")
    size: Optional[str] = Field(default="2048x2048", description="Image resolution. Supports '2K', '4K' for automatic aspect ratio, or specific dimensions like '2048x2048', '1280x720'.")

@mcp.tool()
async def generate_image(params: GenerateImageInput) -> str:
    """
    Generate images using the Volcengine (Jimeng) API.
    
    Capabilities:
    1. **Text-to-Image (T2I)**: Provide only `prompt`.
    2. **Image-to-Image (I2I)**: Provide `prompt` and one reference image in `image_urls`.
    3. **Multi-Image Fusion**: Provide `prompt` and multiple reference images in `image_urls` (only supported by specific models like seedream-4.5).
    
    Image Input Format:
    - **URL**: A publicly accessible HTTP/HTTPS URL.
    - **Base64**: A data URI string, e.g., `data:image/jpeg;base64,...`.
    - **Local Path**: Absolute path to a local image file (e.g., `E:/photos/test.jpg`). The server will automatically convert it.
    """
    endpoint = "/images/generations"
    
    payload = {
        "model": params.model,
        "prompt": params.prompt,
        "size": params.size
    }

    if params.image_urls:
        payload["image"] = [_process_image_input(url) for url in params.image_urls]

    try:
        response = await _make_api_request("POST", endpoint, json_data=payload)
        
        # Format output
        data = response.get("data", [])
        if not data:
            return "No image data returned."
        
        output = f"Image Generation Successful (Model: {params.model})\n\n"
        for i, img in enumerate(data):
            output += f"Image {i+1}: {img.get('url')}\n"
            
        return output
    except Exception as e:
        return f"Error generating image: {str(e)}"

# --- Video Generation ---

class GenerateVideoInput(BaseModel):
    prompt: Optional[str] = Field(default=None, description="Text prompt for video generation. Required for T2V, optional for I2V.")
    model: Optional[str] = Field(default=DEFAULT_VIDEO_MODEL, description="Model ID to use (e.g., 'doubao-seedance-1.5-pro-251215').")
    image_urls: Optional[List[str]] = Field(default=None, description="List of input images. Supports HTTP URLs, Base64 strings, or Local File Paths. \n- **1 Image**: Generates video starting from this First Frame.\n- **2 Images**: Generates video interpolating between First and Last Frame.\n- **Empty**: Performs Text-to-Video generation.")
    ratio: Optional[str] = Field(default="16:9", description="Video aspect ratio (e.g., '16:9', '9:16', '1:1', '4:3').")
    duration: Optional[int] = Field(default=5, description="Target video duration in seconds (usually 2-5s depending on model).")

@mcp.tool()
async def generate_video(params: GenerateVideoInput) -> str:
    """
    Create a video generation task. Supports Text-to-Video and Image-to-Video.
    
    Modes (Automatically detected based on `image_urls`):
    1. **Text-to-Video**: Leave `image_urls` empty.
    2. **First Frame I2V**: Provide exactly 1 image. The video will start with this image.
    3. **First & Last Frame I2V**: Provide exactly 2 images. The video will generate a transition from the first to the second image.
    
    You can provide local file paths (e.g., "C:/images/start.jpg"), and they will be automatically converted to Base64 for the API.
    
    Returns:
    - A **Task ID** string. You must use `get_video_task_result` to query the actual video URL after some time.
    """
    endpoint = "/contents/generations/tasks"
    
    content_list = []
    
    # Add text content if provided
    if params.prompt:
        content_list.append({
            "type": "text",
            "text": params.prompt
        })
    
    # Add image content with automatic role assignment
    if params.image_urls:
        processed_urls = [_process_image_input(url) for url in params.image_urls]
        
        if len(processed_urls) == 1:
            content_list.append({
                "type": "image_url",
                "image_url": {"url": processed_urls[0]},
                "role": "first_frame"
            })
        elif len(processed_urls) == 2:
            content_list.append({
                "type": "image_url",
                "image_url": {"url": processed_urls[0]},
                "role": "first_frame"
            })
            content_list.append({
                "type": "image_url",
                "image_url": {"url": processed_urls[1]},
                "role": "last_frame"
            })
        else:
            return "Error: strict automatic role assignment only supports 1 (First Frame) or 2 (First+Last Frame) images."

    payload = {
        "model": params.model,
        "content": content_list,
        "ratio": params.ratio,
        "duration": params.duration
    }
    
    try:
        response = await _make_api_request("POST", endpoint, json_data=payload)
        task_id = response.get("id")
        return f"Video generation task submitted successfully.\nTask ID: {task_id}\n\nPlease use `get_video_task_result(task_id='{task_id}')` to check the result."
    except Exception as e:
        return f"Error creating video task: {str(e)}"

# --- Video Result Query ---

class GetVideoResultInput(BaseModel):
    task_id: str = Field(..., description="The unique Task ID returned by `generate_video`.")

@mcp.tool()
async def get_video_task_result(params: GetVideoResultInput) -> str:
    """
    Query the status and result of a video generation task.
    
    Usage:
    - Call this periodically after submitting a video task.
    - Status values: 'queued', 'running', 'succeeded', 'failed'.
    - If 'succeeded', returns the final Video URL.
    """
    endpoint = f"/contents/generations/tasks/{params.task_id}"
    
    try:
        response = await _make_api_request("GET", endpoint)
        
        status = response.get("status")
        
        if status == "succeeded":
            content = response.get("content", {})
            video_url = content.get("video_url")
            return f"Task Status: {status}\nVideo URL: {video_url}"
        elif status == "failed":
            error = response.get("error", {})
            return f"Task Status: failed\nError: {error.get('message', 'Unknown error')}"
        else:
            return f"Task Status: {status}"
            
    except Exception as e:
        return f"Error retrieving task result: {str(e)}"

def main():
    mcp.run()

if __name__ == "__main__":
    main()
