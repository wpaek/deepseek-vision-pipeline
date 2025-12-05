"""Q/A synthesis using LLM."""

import torch
from typing import Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer


class QASynthesizer:
    """Generate question-answer-reasoning pairs."""
    
    def __init__(self, model_name: str, config: Dict):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = config
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _create_prompt(self, features: Dict, bin_type: str) -> str:
        """Create prompt based on bin type."""
        prompts = {
            'A': f"Given an image with text: '{features['ocr_text'][:200]}', create a question about the text content.",
            'B': f"Given an image with objects: {', '.join(features['objects'][:5])}, create a question about spatial relationships.",
            'C': f"Given an image described as: '{features['caption']}', create a commonsense reasoning question."
        }
        
        base = prompts.get(bin_type, prompts['C'])
        return f"{base}\n\nProvide:\nQuestion:\nAnswer:\nReasoning:"
    
    def generate(self, features: Dict, bin_type: str) -> Dict[str, str]:
        """Generate Q/A for single image."""
        prompt = self._create_prompt(features, bin_type)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.get('max_tokens', 512),
                temperature=self.config.get('temperature', 0.7),
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._parse_response(response)
    
    def _parse_response(self, response: str) -> Dict[str, str]:
        """Parse LLM response into Q/A/R."""
        lines = response.split('\n')
        result = {'question': '', 'answer': '', 'reasoning': ''}
        
        current_key = None
        for line in lines:
            line = line.strip()
            if line.startswith('Question:'):
                current_key = 'question'
                result[current_key] = line.replace('Question:', '').strip()
            elif line.startswith('Answer:'):
                current_key = 'answer'
                result[current_key] = line.replace('Answer:', '').strip()
            elif line.startswith('Reasoning:'):
                current_key = 'reasoning'
                result[current_key] = line.replace('Reasoning:', '').strip()
            elif current_key and line:
                result[current_key] += ' ' + line
        
        return result
