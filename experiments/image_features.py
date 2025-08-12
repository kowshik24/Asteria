"""
Image Feature Extraction for Asteria Research
Support for CLIP, DINO, and other vision models
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from typing import List, Union, Tuple

class ImageFeatureExtractor:
    """Extract dense features from images using pre-trained models"""
    
    def __init__(self, model_name='clip', device='auto', cache_dir='./models'):
        # Auto-detect best device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"ImageFeatureExtractor using device: {self.device}")
        
        self.model_name = model_name
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        if model_name == 'clip':
            self.model, self.preprocess = self._load_clip()
        elif model_name == 'dino':
            self.model, self.preprocess = self._load_dino()
        elif model_name == 'resnet':
            self.model, self.preprocess = self._load_resnet()
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def _load_clip(self):
        """Load CLIP model"""
        try:
            import clip
            model, preprocess = clip.load("ViT-B/32", device=self.device, download_root=self.cache_dir)
            return model, preprocess
        except ImportError:
            print("CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")
            return self._load_resnet()
    
    def _load_dino(self):
        """Load DINO model"""
        try:
            # Using torchvision's DINO implementation
            model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=True)
            model = model.to(self.device).eval()
            
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            return model, preprocess
        except Exception as e:
            print(f"DINO not available: {e}. Falling back to ResNet.")
            return self._load_resnet()
    
    def _load_resnet(self):
        """Load ResNet50 as fallback"""
        from torchvision import models
        
        model = models.resnet50(pretrained=True)
        # Remove the final classification layer to get features
        model = nn.Sequential(*list(model.children())[:-1])
        model = model.to(self.device).eval()
        
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return model, preprocess
    
    def extract_from_paths(self, image_paths: List[str], batch_size: int = 32) -> np.ndarray:
        """Extract features from image file paths"""
        features = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            
            for path in batch_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    img_tensor = self.preprocess(img)
                    batch_images.append(img_tensor)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    continue
            
            if batch_images:
                batch_tensor = torch.stack(batch_images).to(self.device)
                batch_features = self._extract_batch(batch_tensor)
                features.append(batch_features.cpu().numpy())
        
        return np.vstack(features) if features else np.array([])
    
    def extract_from_tensors(self, image_tensors: torch.Tensor) -> np.ndarray:
        """Extract features from image tensors"""
        return self._extract_batch(image_tensors).cpu().numpy()
    
    @torch.no_grad()
    def _extract_batch(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        """Extract features from a batch of preprocessed images"""
        if self.model_name == 'clip':
            features = self.model.encode_image(batch_tensor)
        elif self.model_name == 'dino':
            features = self.model(batch_tensor)
        elif self.model_name == 'resnet':
            features = self.model(batch_tensor)
            features = features.flatten(1)  # Flatten spatial dimensions
        
        return features

class HierarchicalImageHash:
    """Novel: Multi-scale hashing for images at different resolutions"""
    
    def __init__(self, dim: int, scales: List[int] = [16, 32, 64], device='cpu'):
        from asteria.ecvh import ECVH
        self.scales = scales
        self.device = device
        
        # Create hashers for different scales
        self.hashers = {}
        for scale in scales:
            vantages = min(scale * 2, 128)  # Adaptive vantages
            self.hashers[scale] = ECVH(dim, vantages, scale, scale).to(device)
    
    def encode_hierarchical(self, features: torch.Tensor) -> dict:
        """Encode features at multiple hash lengths"""
        codes = {}
        features = features.to(self.device)
        
        for scale in self.scales:
            with torch.no_grad():
                codes[f'scale_{scale}'] = self.hashers[scale](features)
        
        return codes
    
    def search_hierarchical(self, query_codes: dict, db_codes: dict, k: int = 10) -> dict:
        """Search using hierarchical codes (coarse to fine)"""
        results = {}
        
        # Start with coarsest scale for initial filtering
        for scale in sorted(self.scales):
            scale_key = f'scale_{scale}'
            if scale_key in query_codes and scale_key in db_codes:
                # Implement hamming distance search for this scale
                results[scale_key] = self._hamming_search(
                    query_codes[scale_key], 
                    db_codes[scale_key], 
                    k
                )
        
        return results
    
    def _hamming_search(self, query_codes, db_codes, k):
        """Basic hamming distance search"""
        # Simplified implementation - expand based on your needs
        distances = torch.cdist(query_codes.float(), db_codes.float(), p=0)
        _, indices = torch.topk(distances, k, largest=False)
        return indices

def create_synthetic_image_dataset(num_images: int = 10000, 
                                 image_size: Tuple[int, int] = (224, 224),
                                 num_clusters: int = 100) -> torch.Tensor:
    """Create synthetic image-like data with clustering structure"""
    
    # Create cluster centers
    cluster_centers = torch.randn(num_clusters, *image_size)
    
    # Assign images to clusters
    cluster_assignments = torch.randint(0, num_clusters, (num_images,))
    
    images = torch.zeros(num_images, *image_size)
    
    for i in range(num_images):
        cluster_id = cluster_assignments[i]
        # Add noise to cluster center
        noise = torch.randn(*image_size) * 0.3
        images[i] = cluster_centers[cluster_id] + noise
    
    # Normalize to [0, 1] range
    images = (images - images.min()) / (images.max() - images.min())
    
    return images, cluster_assignments

if __name__ == "__main__":
    # Test the feature extractor
    extractor = ImageFeatureExtractor('resnet', device='cpu')
    
    # Create some test synthetic images
    test_images, _ = create_synthetic_image_dataset(100, (224, 224))
    test_images = test_images.unsqueeze(1).repeat(1, 3, 1, 1)  # Add RGB channels
    
    features = extractor.extract_from_tensors(test_images)
    print(f"Extracted features shape: {features.shape}")
    
    # Test hierarchical hashing
    hierarchical = HierarchicalImageHash(features.shape[1], [16, 32, 64])
    codes = hierarchical.encode_hierarchical(torch.tensor(features))
    
    for scale, code in codes.items():
        print(f"{scale}: {code.shape}")
