"""Main pipeline orchestrator."""

import json
import yaml
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

from deepseek_extractor import DeepSeekExtractor
from binner import ImageBinner
from synthesizer import QASynthesizer


class DeepSeekPipeline:
    """End-to-end vision-language data synthesis pipeline."""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.extractor = DeepSeekExtractor(
            model_name=self.config['model']['name'],
            device=self.config['model']['device'],
            model_size=self.config['model']['size']
        )
        
        self.binner = ImageBinner(self.extractor, self.config['binning'])
        self.synthesizer = QASynthesizer(
            self.config['synthesis']['llm_model'],
            self.config['synthesis']
        )
    
    def load_images(self, images_dir: str, max_images: int = None) -> List[str]:
        """Load image paths from directory."""
        image_dir = Path(images_dir)
        extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        
        images = []
        for ext in extensions:
            images.extend(image_dir.rglob(f'*{ext}'))
        
        images = [str(p) for p in images]
        if max_images:
            images = images[:max_images]
        
        return images
    
    def run(self, images_dir: str, output_dir: str, num_images: int = None):
        """Run full pipeline."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Stage 1: Load images
        print("Loading images...")
        images = self.load_images(images_dir, num_images)
        print(f"Loaded {len(images)} images")
        
        # Stage 2: Bin images
        print("\nBinning images...")
        bins = self.binner.bin_images(tqdm(images, desc="Binning"))
        
        # Balance bins
        bins = self.binner.balance_bins(
            bins,
            self.config['binning']['bins_ratio']
        )
        
        print(f"Binned: A={len(bins['A'])}, B={len(bins['B'])}, C={len(bins['C'])}")
        
        # Save binned results
        if self.config['output']['save_intermediate']:
            self._save_bins(bins, output_path / "bins.jsonl")
        
        # Stage 3: Generate Q/A
        print("\nGenerating Q/A pairs...")
        dataset = []
        
        for bin_type, images in bins.items():
            for img_data in tqdm(images, desc=f"Bin {bin_type}"):
                qa = self.synthesizer.generate(
                    img_data['features'],
                    bin_type
                )
                
                if self._validate(qa):
                    dataset.append({
                        'image': img_data['path'],
                        'bin': bin_type,
                        **qa
                    })
        
        print(f"Generated {len(dataset)} Q/A pairs")
        
        # Save final dataset
        output_file = output_path / f"dataset.{self.config['output']['format']}"
        self._save_dataset(dataset, output_file)
        print(f"\nSaved to: {output_file}")
        
        return dataset
    
    def _validate(self, qa: Dict) -> bool:
        """Validate Q/A quality."""
        val_config = self.config['validation']
        
        q_len = len(qa['question'].split())
        a_len = len(qa['answer'].split())
        r_len = len(qa['reasoning'].split())
        
        return (q_len >= val_config['min_question_length'] and
                a_len >= val_config['min_answer_length'] and
                r_len >= val_config['min_reasoning_length'])
    
    def _save_bins(self, bins: Dict, path: Path):
        """Save binned images."""
        with open(path, 'w') as f:
            for bin_type, images in bins.items():
                for img in images:
                    f.write(json.dumps({
                        'path': img['path'],
                        'bin': bin_type
                    }) + '\n')
    
    def _save_dataset(self, dataset: List[Dict], path: Path):
        """Save final dataset."""
        with open(path, 'w') as f:
            for item in dataset:
                f.write(json.dumps(item) + '\n')
