#!/usr/bin/env python3
"""
Measurement Feature Extractor
Extracts 256-dimensional feature vectors to match the measurement index.
Since the original extraction method is unknown, this provides multiple approaches.
"""

import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import logging
from typing import Optional, Union
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MeasurementFeatureExtractor:
    """
    Extracts 256-dimensional features for the measurement index.
    Supports multiple extraction methods since the original is unknown.
    """
    
    def __init__(self, method: str = "resnet_pool"):
        """
        Initialize the feature extractor
        
        Args:
            method: Feature extraction method
                   - "resnet_pool": ResNet features with pooling to 256d
                   - "combined_cv": Combined computer vision features
                   - "learned_projection": Pre-trained model with learned projection
        """
        self.method = method
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = None
        self.is_loaded = False
        
        logger.info(f"Initializing MeasurementFeatureExtractor with method: {method}")
    
    def load_model(self) -> bool:
        """Load the feature extraction model"""
        try:
            if self.method == "resnet_pool":
                return self._load_resnet_model()
            elif self.method == "combined_cv":
                return self._load_combined_cv_model()
            elif self.method == "learned_projection":
                return self._load_learned_projection_model()
            else:
                logger.error(f"Unknown extraction method: {self.method}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading measurement feature extractor: {e}")
            return False
    
    def _load_resnet_model(self) -> bool:
        """Load ResNet-based feature extractor"""
        try:
            logger.info("Loading ResNet-based feature extractor...")
            
            # Load pre-trained ResNet
            resnet = models.resnet50(pretrained=True)
            
            # Remove the final classification layer
            self.model = nn.Sequential(*list(resnet.children())[:-1])  # Remove avgpool and fc
            
            # Add custom layers to get exactly 256 dimensions
            self.projection = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
                nn.Flatten(),
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                L2Normalize(dim=1)  # L2 normalization for better similarity search
            )
            
            self.model = self.model.to(self.device)
            self.projection = self.projection.to(self.device)
            self.model.eval()
            self.projection.eval()
            
            # Define image preprocessing
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            self.is_loaded = True
            logger.info("✅ ResNet-based feature extractor loaded")
            return True
            
        except Exception as e:
            logger.error(f"Error loading ResNet model: {e}")
            return False
    
    def _load_combined_cv_model(self) -> bool:
        """Load combined computer vision features extractor"""
        try:
            logger.info("Loading combined CV feature extractor...")
            
            # This method combines multiple traditional CV features
            # No PyTorch model needed for this approach
            self.is_loaded = True
            logger.info("✅ Combined CV feature extractor loaded")
            return True
            
        except Exception as e:
            logger.error(f"Error loading combined CV model: {e}")
            return False
    
    def _load_learned_projection_model(self) -> bool:
        """Load a model with learned projection (if available)"""
        try:
            logger.info("Loading learned projection model...")
            
            # Check if there's a saved projection model
            projection_path = "indexes/index_measurements/projection_model.pth"
            if os.path.exists(projection_path):
                logger.info(f"Found saved projection model at {projection_path}")
                # Load the saved model
                # This would be specific to how the measurement index was originally created
                pass
            
            # For now, fall back to ResNet method
            logger.warning("No learned projection model found, falling back to ResNet method")
            return self._load_resnet_model()
            
        except Exception as e:
            logger.error(f"Error loading learned projection model: {e}")
            return False
    
    def extract_features(self, image_path: str) -> Optional[np.ndarray]:
        """
        Extract 256-dimensional features from an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            256-dimensional feature vector or None if extraction fails
        """
        if not self.is_loaded:
            logger.error("Feature extractor not loaded")
            return None
        
        try:
            if self.method == "resnet_pool":
                return self._extract_resnet_features(image_path)
            elif self.method == "combined_cv":
                return self._extract_combined_cv_features(image_path)
            elif self.method == "learned_projection":
                return self._extract_learned_projection_features(image_path)
            else:
                logger.error(f"Unknown extraction method: {self.method}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting features from {image_path}: {e}")
            return None
    
    def _extract_resnet_features(self, image_path: str) -> Optional[np.ndarray]:
        """Extract features using ResNet + projection"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Extract ResNet features
                features = self.model(image_tensor)
                
                # Project to 256 dimensions
                projected_features = self.projection(features)
                
                # Convert to numpy
                feature_vector = projected_features.cpu().numpy().flatten()
                
                return feature_vector.astype(np.float32)
                
        except Exception as e:
            logger.error(f"Error in ResNet feature extraction: {e}")
            return None
    
    def _extract_combined_cv_features(self, image_path: str) -> Optional[np.ndarray]:
        """Extract combined computer vision features"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return None
            
            # Resize to standard size
            image = cv2.resize(image, (224, 224))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            features = []
            
            # 1. Color histogram features (64 dimensions)
            hist_b = cv2.calcHist([image], [0], None, [16], [0, 256])
            hist_g = cv2.calcHist([image], [1], None, [16], [0, 256])
            hist_r = cv2.calcHist([image], [2], None, [16], [0, 256])
            color_features = np.concatenate([hist_b.flatten(), hist_g.flatten(), hist_r.flatten()])
            color_features = color_features / (np.linalg.norm(color_features) + 1e-8)  # Normalize
            features.extend(color_features[:64])
            
            # 2. Texture features using LBP (64 dimensions)
            from skimage.feature import local_binary_pattern
            radius = 3
            n_points = 8 * radius
            lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=64, range=(0, 64))
            lbp_hist = lbp_hist / (np.linalg.norm(lbp_hist) + 1e-8)  # Normalize
            features.extend(lbp_hist)
            
            # 3. Edge features using HOG (64 dimensions)
            from skimage.feature import hog
            hog_features = hog(gray, orientations=8, pixels_per_cell=(16, 16),
                              cells_per_block=(1, 1), visualize=False)
            hog_features = hog_features / (np.linalg.norm(hog_features) + 1e-8)  # Normalize
            # Reduce to 64 dimensions
            if len(hog_features) > 64:
                hog_features = hog_features[:64]
            else:
                hog_features = np.pad(hog_features, (0, 64 - len(hog_features)), 'constant')
            features.extend(hog_features)
            
            # 4. Shape features (64 dimensions)
            # Contour-based features
            contours, _ = cv2.findContours(cv2.Canny(gray, 50, 150), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                moments = cv2.moments(largest_contour)
                
                # Hu moments and other shape features
                hu_moments = cv2.HuMoments(moments).flatten()
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                
                shape_features = np.concatenate([
                    hu_moments,
                    [area / (224 * 224), perimeter / (4 * 224), 
                     area / (perimeter * perimeter + 1e-8)]  # Normalized shape metrics
                ])
                shape_features = np.pad(shape_features, (0, 64 - len(shape_features)), 'constant')
            else:
                shape_features = np.zeros(64)
            
            features.extend(shape_features)
            
            # Ensure exactly 256 dimensions
            feature_vector = np.array(features[:256], dtype=np.float32)
            if len(feature_vector) < 256:
                feature_vector = np.pad(feature_vector, (0, 256 - len(feature_vector)), 'constant')
            
            # Final normalization
            feature_vector = feature_vector / (np.linalg.norm(feature_vector) + 1e-8)
            
            return feature_vector
            
        except Exception as e:
            logger.error(f"Error in combined CV feature extraction: {e}")
            return None
    
    def _extract_learned_projection_features(self, image_path: str) -> Optional[np.ndarray]:
        """Extract features using learned projection (fallback to ResNet for now)"""
        return self._extract_resnet_features(image_path)
    
    def batch_extract_features(self, image_paths: list) -> np.ndarray:
        """Extract features from multiple images in batch"""
        features = []
        
        for image_path in image_paths:
            feature = self.extract_features(image_path)
            if feature is not None:
                features.append(feature)
            else:
                # Use zero vector for failed extractions
                features.append(np.zeros(256, dtype=np.float32))
        
        return np.array(features)
    
    def get_feature_dimension(self) -> int:
        """Get the dimension of extracted features"""
        return 256
    
    def get_method_info(self) -> dict:
        """Get information about the extraction method"""
        return {
            'method': self.method,
            'dimension': 256,
            'device': str(self.device),
            'is_loaded': self.is_loaded
        }

# Custom L2 normalization layer
class L2Normalize(nn.Module):
    def __init__(self, dim=1):
        super(L2Normalize, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        return torch.nn.functional.normalize(x, p=2, dim=self.dim) 