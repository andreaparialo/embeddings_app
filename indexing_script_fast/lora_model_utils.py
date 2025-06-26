#!/usr/bin/env python3
"""
Utility functions for discovering and managing LoRA models
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

def get_available_lora_models(finetuning_dir: str = "finetuning/output_single_gpu") -> List[Dict[str, str]]:
    """
    Discover all available LoRA models with their metadata.
    
    Args:
        finetuning_dir: Directory containing LoRA model outputs
        
    Returns:
        List of dictionaries containing model information
    """
    models = []
    finetuning_output_dir = Path(finetuning_dir)
    
    if not finetuning_output_dir.exists():
        logger.warning(f"Finetuning directory {finetuning_dir} does not exist")
        return models
    
    # Sort by version name (v0, v1, v2, etc.) with latest first
    for model_dir in sorted(finetuning_output_dir.iterdir(), key=lambda x: x.name, reverse=True):
        if model_dir.is_dir() and model_dir.name.startswith("v"):
            checkpoint_dir = model_dir / "checkpoint-112"
            if checkpoint_dir.exists():
                # Load metadata for this model
                args_file = model_dir / "args.json"
                model_info = {
                    'path': str(checkpoint_dir),
                    'version': model_dir.name,
                    'timestamp': model_dir.name.split('-', 1)[1] if '-' in model_dir.name else 'Unknown',
                    'epochs': 'N/A',
                    'learning_rate': 'N/A',
                    'loss_type': 'N/A',
                    'lora_rank': 'N/A',
                    'lora_alpha': 'N/A',
                    'display_name': f"{model_dir.name} ({model_dir.name.split('-', 1)[1] if '-' in model_dir.name else 'Unknown'})"
                }
                
                if args_file.exists():
                    try:
                        with open(args_file, 'r') as f:
                            args = json.load(f)
                        model_info['epochs'] = args.get('num_train_epochs', 'N/A')
                        model_info['learning_rate'] = args.get('learning_rate', 'N/A')
                        model_info['loss_type'] = args.get('loss_type', 'N/A')
                        model_info['lora_rank'] = args.get('lora_rank', 'N/A')
                        model_info['lora_alpha'] = args.get('lora_alpha', 'N/A')
                        
                        # Update display name with more details
                        model_info['display_name'] = f"{model_info['version']} | {model_info['timestamp']} | LR:{model_info['learning_rate']} | Rank:{model_info['lora_rank']}"
                    except Exception as e:
                        logger.warning(f"Could not load args.json for {model_dir}: {e}")
                
                models.append(model_info)
    
    return models

def get_latest_lora_model(finetuning_dir: str = "finetuning/output_single_gpu") -> Optional[str]:
    """
    Get the path to the latest LoRA model (for backward compatibility).
    
    Args:
        finetuning_dir: Directory containing LoRA model outputs
        
    Returns:
        Path to the latest LoRA checkpoint, or None if none found
    """
    models = get_available_lora_models(finetuning_dir)
    if models:
        return models[0]['path']  # First model is the latest due to reverse sorting
    return None

def find_lora_model_by_version(version: str, finetuning_dir: str = "finetuning/output_single_gpu") -> Optional[str]:
    """
    Find a LoRA model by its version identifier.
    
    Args:
        version: Version identifier (e.g., "v9", "v8", etc.)
        finetuning_dir: Directory containing LoRA model outputs
        
    Returns:
        Path to the LoRA checkpoint, or None if not found
    """
    models = get_available_lora_models(finetuning_dir)
    for model in models:
        if model['version'] == version:
            return model['path']
    return None

def list_lora_models_summary(finetuning_dir: str = "finetuning/output_single_gpu") -> None:
    """
    Print a summary of available LoRA models to console.
    
    Args:
        finetuning_dir: Directory containing LoRA model outputs
    """
    models = get_available_lora_models(finetuning_dir)
    
    if not models:
        print("❌ No LoRA models found")
        return
    
    print(f"\n🎯 Available LoRA Models ({len(models)} found):")
    print("=" * 80)
    
    for i, model in enumerate(models):
        status = "🏆 Latest" if i == 0 else "📦 Available"
        print(f"{status} {model['display_name']}")
        print(f"    Path: {model['path']}")
        print(f"    Training: {model['epochs']} epochs, Loss: {model['loss_type']}")
        print()

def get_model_comparison_info(models: List[str], finetuning_dir: str = "finetuning/output_single_gpu") -> List[Dict]:
    """
    Get comparison information for multiple models.
    
    Args:
        models: List of version identifiers to compare
        finetuning_dir: Directory containing LoRA model outputs
        
    Returns:
        List of model information dictionaries for comparison
    """
    all_models = get_available_lora_models(finetuning_dir)
    comparison_models = []
    
    for version in models:
        for model in all_models:
            if model['version'] == version:
                comparison_models.append(model)
                break
    
    return comparison_models

if __name__ == "__main__":
    # Demo/test the utility functions
    print("🔍 LoRA Model Discovery Demo")
    list_lora_models_summary()
    
    latest = get_latest_lora_model()
    if latest:
        print(f"\n🏆 Latest model: {latest}")
    
    # Test finding specific version
    v9_model = find_lora_model_by_version("v9")
    if v9_model:
        print(f"🎯 Found v9 model: {v9_model}") 