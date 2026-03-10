import os
import httpx
import base64
import mimetypes
from pathlib import Path
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
VOLC_API_KEY = os.getenv("VOLC_API_KEY")
DEFAULT_IMAGE_MODEL = os.getenv("VOLC_IMAGE_MODEL", "doubao-seedream-4.5")
DEFAULT_VIDEO_MODEL = os.getenv("VOLC_VIDEO_MODEL", "doubao-seedance-1.5-pro-251215")

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
    if image_input.startswith("http") or image_input.startswith("data:"):
        return image_input
    
    file_path = Path(image_input)
    if file_path.exists() and file_path.is_file():
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            suffix = file_path.suffix.lower()
            if suffix in ['.jpg', '.jpeg']: mime_type = 'image/jpeg'
            elif suffix == '.png': mime_type = 'image/png'
            elif suffix == '.webp': mime_type = 'image/webp'
            else: mime_type = 'application/octet-stream'
            
        with open(file_path, "rb") as f:
            encoded_string = base64.b64encode(f.read()).decode('utf-8')
            return f"data:{mime_type};base64,{encoded_string}"
            
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
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.request(method, url, headers=headers, json=json_data, params=params)
        
        if response.status_code >= 400:
            error_msg = f"API Error {response.status_code}: {response.text}"
            try:
                error_data = response.json()
                if "error" in error_data:
                    error_msg = f"API Error {response.status_code} ({error_data['error'].get('code', 'unknown')}): {error_data['error'].get('message', response.text)}"
            except:
                pass
            raise httpx.HTTPStatusError(error_msg, request=response.request, response=response)
        
        return response.json()

# --- Image Generation ---

# Maps ratio strings to pixel dimensions for models requiring specific sizes
IMAGE_SIZE_MAP = {
    "1:1":  "2048x2048",
    "4:3":  "2304x1728",
    "3:4":  "1728x2304",
    "16:9": "2560x1440",
    "9:16": "1440x2560",
    "3:2":  "2496x1664",
    "2:3":  "1664x2496",
    "21:9": "3024x1296",
}

class GenerateImageInput(BaseModel):
    prompt: str = Field(..., description="Text prompt describing the desired image. Required for both Text-to-Image and Image-to-Image.")
    model: Optional[str] = Field(default=DEFAULT_IMAGE_MODEL, description="[CRITICAL] Leave this entirely EMPTY/OMITTED unless the user explicitly asks to use a custom Endpoint ID or model. The server already uses the configured default API Endpoint.")
    image_urls: Optional[List[str]] = Field(default=None, description="List of reference images for Image-to-Image (URL, Base64, or local absolute path). Example: ['https://example.com/img.png']")
    size: Optional[str] = Field(default="1:1", description="[CRITICAL] Image resolution/proportion. Accepted values exactly: '1:1', '16:9', '4:3', '9:16', '3:2', '2:3', '21:9', '2K', '4K'. DO NOT use 'ratio' parameter for images.")
    seed: Optional[int] = Field(default=-1, description="Random seed for reproducibility (-1 for random).")
    response_format: Optional[str] = Field(default="url", description="Format of the returned image: 'url' (recommended) or 'b64_json'.")

@mcp.tool()
async def generate_image(params: GenerateImageInput) -> str:
    """
    Generate images using Volcengine visual generation API.
    
    CRITICAL INSTRUCTIONS FOR AI AGENTS:
    - For image size, ONLY use the 'size' parameter (e.g., size="16:9"). DO NOT pass a 'ratio' or 'width'/'height' parameter.
    - If the user provides a custom Endpoint ID (starts with 'ep-'), you MUST pass it into the 'model' parameter.
    
    Capabilities:
    1. Text-to-Image: Provide 'prompt' and 'size'.
    2. Image-to-Image: Provide 'prompt' AND 'image_urls' (list containing 1 image string).
    """
    endpoint = "/images/generations"
    
    # Resolve size if it's a known ratio key
    resolved_size = IMAGE_SIZE_MAP.get(params.size, params.size)
    
    payload = {
        "model": params.model,
        "prompt": params.prompt,
        "size": resolved_size,
        "seed": params.seed,
        "response_format": params.response_format
    }

    if params.image_urls:
        processed_urls = [_process_image_input(url) for url in params.image_urls]
        payload["image"] = processed_urls # Recent API supports list for multi-image

    try:
        response = await _make_api_request("POST", endpoint, json_data=payload)
        data = response.get("data", [])
        
        output = f"Image Generation Successful (Model: {params.model}, Size: {resolved_size})\n"
        for i, img in enumerate(data):
            if params.response_format == "url":
                output += f"\nImage {i+1}: {img.get('url')}\n"
            else:
                output += f"\nImage {i+1}: [Base64 Data Omitted]\n"
        return output
    except Exception as e:
        return f"Error generating image: {str(e)}"

# --- Video Generation ---

class GenerateVideoInput(BaseModel):
    prompt: Optional[str] = Field(default=None, description="Text prompt for the video. Required for Text-to-Video. Optional/supplementary for Image-to-Video.")
    model: Optional[str] = Field(default=DEFAULT_VIDEO_MODEL, description="[CRITICAL] Leave this entirely EMPTY/OMITTED unless the user explicitly asks to use a custom Endpoint ID/model. The server already uses the configured default API Endpoint.")
    image_urls: Optional[List[str]] = Field(default=None, description="[Image-to-Video] Input images. Pro models: 1 image (First Frame) or 2 images (First+Last Frame). Lite models: 1-4 images (Multi-Image Fusion).")
    ratio: Optional[str] = Field(default="16:9", description="[CRITICAL] Video aspect ratio. Accepted values exactly: '16:9', '9:16', '1:1', '4:3', '3:4', '21:9'. DO NOT use 'size' parameter here.")
    resolution: Optional[str] = Field(default="720p", description="Video resolution. Accepted values exactly: '720p', '1080p'.")
    duration: Optional[int] = Field(default=5, description="Video duration in seconds. Supported range depends on model (usually 4-12s, default 5).")
    seed: Optional[int] = Field(default=-1, description="Random seed for reproducibility (-1 for random).")
    generate_audio: bool = Field(default=True, description="Whether to generate an audio track (only supported by Pro models).")
    watermark: bool = Field(default=False, description="Whether to add a watermark to the generated video.")

@mcp.tool()
async def generate_video(params: GenerateVideoInput) -> str:
    """
    Create a video generation task using Volcengine API. 
    
    CRITICAL INSTRUCTIONS FOR AI AGENTS:
    - This tool ONLY submits the task. It returns a Task ID. You MUST subsequently call `get_video_task_result` in a loop (wait 5-10s between calls) to retrieve the actual video URL.
    - Use 'ratio' for video proportions (e.g., ratio="16:9"). DO NOT use 'size' or 'width'/'height'.
    - If user provides an Endpoint ID ('ep-...'), pass it to the 'model' parameter.
    
    Capabilities:
    1. Text-to-Video: Provide 'prompt' and 'ratio'.
    2. Image-to-Video: Provide 'image_urls' array + optional 'prompt'.
    """
    endpoint = "/contents/generations/tasks"
    
    content_list = []
    if params.prompt:
        content_list.append({"type": "text", "text": params.prompt})
    
    if params.image_urls:
        processed_urls = [_process_image_input(url) for url in params.image_urls]
        
        # Detect model type for role assignment
        # Lite models (e.g., doubao-seedance-1.0-lite-i2v) use 'reference_image' for fusion (1-4 images)
        is_lite = "lite" in (params.model or "").lower()
        
        if is_lite:
            # Lite / Reference Mode: All images are references
            for url in processed_urls:
                content_list.append({"type": "image_url", "image_url": {"url": url}, "role": "reference_image"})
        else:
            # Pro / Standard Mode: First/Last Frame Logic
            if len(processed_urls) == 1:
                content_list.append({"type": "image_url", "image_url": {"url": processed_urls[0]}, "role": "first_frame"})
            elif len(processed_urls) == 2:
                content_list.append({"type": "image_url", "image_url": {"url": processed_urls[0]}, "role": "first_frame"})
                content_list.append({"type": "image_url", "image_url": {"url": processed_urls[1]}, "role": "last_frame"})
            else:
                return "Error: Pro models only support 1 (First Frame) or 2 (First+Last Frame) images. Use a 'lite' model for multi-image fusion (1-4 images)."

    payload = {
        "model": params.model,
        "content": content_list,
        "ratio": params.ratio,
        "resolution": params.resolution,
        "duration": params.duration,
        "seed": params.seed,
        "watermark": params.watermark,
        "generate_audio": params.generate_audio
    }
    
    try:
        response = await _make_api_request("POST", endpoint, json_data=payload)
        task_id = response.get("id")
        return f"Video task submitted successfully.\nTask ID: {task_id}\nUse `get_video_task_result(task_id='{task_id}')` to check status."
    except Exception as e:
        return f"Error creating video task: {str(e)}"

# --- Video Result Query ---

class GetVideoResultInput(BaseModel):
    task_id: str = Field(..., description="Task ID from generate_video.")

@mcp.tool()
async def get_video_task_result(params: GetVideoResultInput) -> str:
    """
    Query the status of a video generation task using the Task ID returned by `generate_video`.
    
    CRITICAL INSTRUCTIONS FOR AI AGENTS:
    - Video generation takes time (often 1-3 minutes).
    - If the returned status is "ordered" or "running", DO NOT tell the user it failed. Ask the user to wait, and call this tool again after 10-15 seconds.
    - Once the status is "succeeded", the response will contain the download URL.
    """
    endpoint = f"/contents/generations/tasks/{params.task_id}"
    try:
        response = await _make_api_request("GET", endpoint)
        status = response.get("status")
        
        if status in ["ordered", "running"]:
            return f"Task Status: {status}\nThe video is still generating. Please query again in a few seconds."
        elif status == "succeeded":
            content = response.get("content", {})
            return f"Task Status: succeeded\nVideo URL: {content.get('video_url', 'No URL found')}"
        elif status == "failed":
            error = response.get("error", {})
            return f"Task Status: failed\nError: {error.get('message', 'Unknown error')}"
        else:
            return f"Task Status: {status}\nFull Response: {response}"
    except Exception as e:
        # Standardize return for CLI viewing
        return f"Error retrieving task result: {str(e)}"

def main():
    mcp.run()

if __name__ == "__main__":
    main()
