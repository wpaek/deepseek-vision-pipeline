"""Image binning based on extracted features."""

from typing import Dict, List
from deepseek_extractor import DeepSeekExtractor


class ImageBinner:
    """Categorize images into Text/Object/Commonsense bins."""
    
    def __init__(self, extractor: DeepSeekExtractor, config: Dict):
        self.extractor = extractor
        self.text_threshold = config.get('text_threshold', 50)
        self.object_threshold = config.get('object_threshold', 5)
    
    def categorize(self, image_path: str) -> tuple:
        """Categorize image and return bin type with features."""
        features = self.extractor.extract_all(image_path)
        
        # Bin A: Text-heavy images
        if len(features['ocr_text']) >= self.text_threshold:
            return 'A', features
        
        # Bin B: Object-rich images
        if len(features['objects']) >= self.object_threshold:
            return 'B', features
        
        # Bin C: Commonsense/general images
        return 'C', features
    
    def bin_images(self, image_paths: List[str]) -> Dict[str, List[Dict]]:
        """Bin multiple images."""
        bins = {'A': [], 'B': [], 'C': []}
        
        for path in image_paths:
            bin_type, features = self.categorize(path)
            bins[bin_type].append({
                'path': path,
                'features': features
            })
        
        return bins
    
    def balance_bins(self, bins: Dict[str, List[Dict]], 
                     ratio: List[float]) -> Dict[str, List[Dict]]:
        """Balance bins according to target ratio."""
        total = sum(len(b) for b in bins.values())
        target_a = int(total * ratio[0])
        target_b = int(total * ratio[1])
        target_c = total - target_a - target_b
        
        return {
            'A': bins['A'][:target_a],
            'B': bins['B'][:target_b],
            'C': bins['C'][:target_c]
        }
