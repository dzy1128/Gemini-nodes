"""
Gemini Image Generation Node for ComfyUI
"""

import os
import time
import torch
import numpy as np
from PIL import Image
import io
import base64
import mimetypes

from google import genai
from google.genai import types


class GeminiImageGenerator:
    """
    A ComfyUI node that uses Google Gemini API to generate images.
    Supports up to 5 optional input images and various configuration options.
    """
    
    MODELS = [
        "gemini-3-pro-image-preview",
    ]
    
    ASPECT_RATIOS = [
        "1:1",
        "16:9",
        "9:16",
        "4:3",
        "3:4",
        "3:2",
        "2:3",
        "21:9",
        "9:21",
    ]
    
    IMAGE_SIZES = [
        "1K",
        "2K",
        "4K",
    ]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (cls.MODELS, {
                    "default": cls.MODELS[0]
                }),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Enter your prompt here..."
                }),
                "aspect_ratio": (cls.ASPECT_RATIOS, {
                    "default": "1:1"
                }),
                "image_size": (cls.IMAGE_SIZES, {
                    "default": "1K"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "step": 1,
                    "display": "number"
                }),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("text", "image")
    FUNCTION = "generate"
    CATEGORY = "Gemini"
    OUTPUT_NODE = False
    
    def tensor_to_pil(self, tensor):
        """Convert a ComfyUI tensor to PIL Image"""
        # ComfyUI uses (B, H, W, C) format with values in [0, 1]
        if tensor is None:
            return None
        
        # Take the first image if batch
        if len(tensor.shape) == 4:
            tensor = tensor[0]
        
        # Convert to numpy and scale to 0-255
        np_image = (tensor.cpu().numpy() * 255).astype(np.uint8)
        
        return Image.fromarray(np_image)
    
    def pil_to_tensor(self, pil_image):
        """Convert a PIL Image to ComfyUI tensor"""
        # Convert to numpy array
        np_image = np.array(pil_image).astype(np.float32) / 255.0
        
        # Ensure RGB
        if len(np_image.shape) == 2:
            np_image = np.stack([np_image] * 3, axis=-1)
        elif np_image.shape[-1] == 4:
            np_image = np_image[:, :, :3]
        
        # Convert to tensor (B, H, W, C)
        tensor = torch.from_numpy(np_image).unsqueeze(0)
        
        return tensor
    
    def encode_image_to_base64(self, pil_image):
        """Encode PIL Image to base64 string"""
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def create_image_part(self, pil_image):
        """Create a Gemini Part from PIL Image"""
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        
        return types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/png"
        )
    
    def generate(self, model, prompt, aspect_ratio, image_size, seed,
                 image_1=None, image_2=None, image_3=None, image_4=None, image_5=None):
        """
        Generate image using Gemini API
        
        Args:
            model: Model name to use
            prompt: Text prompt for generation
            aspect_ratio: Desired aspect ratio
            image_size: Output image size (1K, 2K, 4K)
            seed: Seed value (not passed to API, used for cache control)
            image_1-5: Optional input images
            
        Returns:
            tuple: (info_text, generated_image_tensor)
        """
        
        # Track timing
        start_time = time.time()
        encode_time = 0
        decode_time = 0
        api_time = 0
        
        # Get API key from environment variable
        api_key = os.environ.get("GEMINI_API_KEY", "")
        
        if not api_key:
            error_msg = "Error: GEMINI_API_KEY environment variable is not set. Please set it before using this node."
            # Return a placeholder black image
            placeholder = torch.zeros((1, 512, 512, 3))
            return (error_msg, placeholder)
        
        try:
            # Initialize client
            client = genai.Client(api_key=api_key)
            
            # Build content parts
            parts = []
            
            # Process input images
            input_images = [image_1, image_2, image_3, image_4, image_5]
            image_count = 0
            
            encode_start = time.time()
            for idx, img_tensor in enumerate(input_images):
                if img_tensor is not None:
                    pil_img = self.tensor_to_pil(img_tensor)
                    if pil_img:
                        parts.append(self.create_image_part(pil_img))
                        image_count += 1
            encode_time = time.time() - encode_start
            
            # Add text prompt
            if prompt.strip():
                parts.append(types.Part.from_text(text=prompt))
            elif image_count == 0:
                error_msg = "Error: Please provide at least a prompt or an input image."
                placeholder = torch.zeros((1, 512, 512, 3))
                return (error_msg, placeholder)
            
            # Create content
            contents = [
                types.Content(
                    role="user",
                    parts=parts,
                ),
            ]
            
            # Configure generation
            generate_content_config = types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
                image_config=types.ImageConfig(
                    image_size=image_size,
                ),
            )
            
            # Call API
            api_start = time.time()
            
            generated_image = None
            response_text = ""
            
            for chunk in client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=generate_content_config,
            ):
                if (
                    chunk.candidates is None
                    or chunk.candidates[0].content is None
                    or chunk.candidates[0].content.parts is None
                ):
                    continue
                
                for part in chunk.candidates[0].content.parts:
                    if part.inline_data and part.inline_data.data:
                        # Decode image
                        decode_start = time.time()
                        image_data = part.inline_data.data
                        pil_image = Image.open(io.BytesIO(image_data))
                        
                        # Convert RGBA to RGB if necessary
                        if pil_image.mode == 'RGBA':
                            pil_image = pil_image.convert('RGB')
                        
                        generated_image = self.pil_to_tensor(pil_image)
                        decode_time = time.time() - decode_start
                    elif part.text:
                        response_text += part.text
            
            api_time = time.time() - api_start
            total_time = time.time() - start_time
            
            # Build result info
            if generated_image is not None:
                img_height, img_width = generated_image.shape[1], generated_image.shape[2]
                
                info_lines = [
                    "✅ Image Generation Successful",
                    f"",
                    f"📐 Output Size: {img_width} x {img_height}",
                    f"📏 Aspect Ratio: {aspect_ratio}",
                    f"🖼️ Image Size Setting: {image_size}",
                    f"📸 Input Images: {image_count}",
                    f"",
                    f"⏱️ Timing:",
                    f"   • Image Encoding: {encode_time:.3f}s",
                    f"   • API Generation: {api_time:.3f}s",
                    f"   • Image Decoding: {decode_time:.3f}s",
                    f"   • Total Time: {total_time:.3f}s",
                ]
                
                if response_text:
                    info_lines.extend([
                        f"",
                        f"📝 API Response:",
                        response_text
                    ])
                
                info_text = "\n".join(info_lines)
                
                return (info_text, generated_image)
            else:
                error_msg = f"❌ Image Generation Failed\n\nNo image was generated by the API.\n"
                if response_text:
                    error_msg += f"\nAPI Response:\n{response_text}"
                
                error_msg += f"\n\n⏱️ API Time: {api_time:.3f}s"
                
                # Return placeholder
                placeholder = torch.zeros((1, 512, 512, 3))
                return (error_msg, placeholder)
                
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}\n\nPlease check your API key and network connection."
            placeholder = torch.zeros((1, 512, 512, 3))
            return (error_msg, placeholder)


# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "GeminiImageGenerator": GeminiImageGenerator,
}

# Display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiImageGenerator": "Gemini Image Generator",
}

