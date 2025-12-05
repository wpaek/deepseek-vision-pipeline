"""DeepSeek-OCR feature extractor for unified vision tasks."""

import torch
import tempfile
import shutil
from PIL import Image
from typing import Dict, List, Any
from transformers import AutoModel, AutoTokenizer


class DeepSeekExtractor:
    """Extract features using DeepSeek-OCR for all vision tasks."""
    
    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-OCR", 
                 device: str = "cuda", model_size: str = "tiny"):
        self.device = device
        self.model_size = model_size
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        
        self.model = AutoModel.from_pretrained(
            model_name,
            _attn_implementation='flash_attention_2',
            torch_dtype=dtype,
            device_map=device if device != "cpu" else None,
            trust_remote_code=True,
            use_safetensors=True
        ).eval()
        
        size_map = {"tiny": 512, "small": 640, "base": 1024, "large": 1280}
        self.image_size = size_map.get(model_size, 512)
    
    def _load_image(self, image_path: str) -> Image.Image:
        """Load and preprocess image."""
        img = Image.open(image_path).convert('RGB')
        return img.resize((self.image_size, self.image_size))
    
    def extract_text(self, image_path: str) -> str:
        """Extract text using DeepSeek-OCR."""
        temp_dir = tempfile.mkdtemp(prefix='deepseek_ocr_')
        
        try:
            result = self.model.infer(
                self.tokenizer,
                prompt="<image>\n<|grounding|>Convert the document to markdown.",
                image_file=str(image_path),
                output_path=temp_dir,
                base_size=self.image_size,
                image_size=self.image_size,
                crop_mode=False,
                save_results=False,
                test_compress=False
            )
            return result.get('text', '') if isinstance(result, dict) else str(result)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def extract_objects(self, image_path: str) -> List[str]:
        """Detect objects using DeepSeek grounding."""
        temp_dir = tempfile.mkdtemp(prefix='deepseek_ocr_')
        
        try:
            result = self.model.infer(
                self.tokenizer,
                prompt="<image>\nList all objects visible in this image.",
                image_file=str(image_path),
                output_path=temp_dir,
                base_size=self.image_size,
                image_size=self.image_size,
                crop_mode=False,
                save_results=False,
                test_compress=False
            )
            text = result.get('text', '') if isinstance(result, dict) else str(result)
            objects = [obj.strip() for obj in text.split(',') if obj.strip()]
            return objects
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def extract_caption(self, image_path: str) -> str:
        """Generate caption using DeepSeek."""
        temp_dir = tempfile.mkdtemp(prefix='deepseek_ocr_')
        
        try:
            result = self.model.infer(
                self.tokenizer,
                prompt="<image>\nDescribe this image in detail.",
                image_file=str(image_path),
                output_path=temp_dir,
                base_size=self.image_size,
                image_size=self.image_size,
                crop_mode=False,
                save_results=False,
                test_compress=False
            )
            return result.get('text', '') if isinstance(result, dict) else str(result)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def extract_all(self, image_path: str) -> Dict[str, Any]:
        """Extract all features from image."""
        return {
            'ocr_text': self.extract_text(image_path),
            'objects': self.extract_objects(image_path),
            'caption': self.extract_caption(image_path)
        }
