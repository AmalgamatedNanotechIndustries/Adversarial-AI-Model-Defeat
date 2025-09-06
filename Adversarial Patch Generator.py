#!/usr/bin/env python3
"""
Adversarial Patch Generator for AI Camera Evasion
Based on research from PlateShapez, DeepPayload, and academic papers on adversarial attacks

This script generates adversarial patches designed to fool AI-enabled cameras and backends
with support for different color modes and QR code overlays.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import qrcode
import matplotlib.pyplot as plt
import os
import random
from typing import Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

class AdversarialPatchGenerator:
    """
    Generates adversarial patches using various techniques including:
    - Fast Gradient Sign Method (FGSM)
    - Projected Gradient Descent (PGD)
    - Patch-based attacks
    - Color space optimization
    """
    
    def __init__(self, patch_size: Tuple[int, int] = (224, 224), device: str = 'cpu'):
        self.patch_size = patch_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.patch = None
        
    def initialize_patch(self, method: str = 'random') -> torch.Tensor:
        """Initialize adversarial patch with different methods"""
        if method == 'random':
            patch = torch.rand(3, self.patch_size[0], self.patch_size[1])
        elif method == 'noise':
            patch = torch.randn(3, self.patch_size[0], self.patch_size[1]) * 0.1 + 0.5
        elif method == 'gradient':
            # Create gradient pattern
            x = torch.linspace(0, 1, self.patch_size[1])
            y = torch.linspace(0, 1, self.patch_size[0])
            X, Y = torch.meshgrid(x, y, indexing='ij')
            patch = torch.stack([X, Y, X*Y])
        else:
            patch = torch.ones(3, self.patch_size[0], self.patch_size[1]) * 0.5
            
        # Ensure values are in [0, 1]
        patch = torch.clamp(patch, 0, 1)
        return patch.to(self.device)
    
    def fgsm_attack(self, patch: torch.Tensor, target_confidence: float = 0.1, 
                   epsilon: float = 0.3, iterations: int = 100) -> torch.Tensor:
        """
        Fast Gradient Sign Method for patch generation
        Optimizes patch to reduce detection confidence
        """
        patch = patch.clone().detach().requires_grad_(True)
        
        for i in range(iterations):
            # Simulate detection confidence (replace with actual model if available)
            confidence = self._simulate_detection_confidence(patch)
            
            # Calculate loss (we want to minimize detection confidence)
            loss = confidence
            
            # Backward pass
            loss.backward()
            
            # FGSM update
            with torch.no_grad():
                patch_grad = patch.grad.sign()
                patch = patch - epsilon * patch_grad
                patch = torch.clamp(patch, 0, 1)
                
            patch = patch.detach().requires_grad_(True)
            
            if i % 20 == 0:
                print(f"Iteration {i}, Confidence: {confidence.item():.4f}")
                
        return patch.detach()
    
    def pgd_attack(self, patch: torch.Tensor, target_confidence: float = 0.1,
                  epsilon: float = 0.3, alpha: float = 0.01, iterations: int = 200) -> torch.Tensor:
        """
        Projected Gradient Descent for more refined patch generation
        """
        original_patch = patch.clone()
        patch = patch.clone().detach().requires_grad_(True)
        
        for i in range(iterations):
            confidence = self._simulate_detection_confidence(patch)
            loss = confidence
            
            loss.backward()
            
            with torch.no_grad():
                # PGD update
                patch = patch - alpha * patch.grad.sign()
                
                # Project back to epsilon ball
                eta = patch - original_patch
                eta = torch.clamp(eta, -epsilon, epsilon)
                patch = torch.clamp(original_patch + eta, 0, 1)
                
            patch = patch.detach().requires_grad_(True)
            
            if i % 40 == 0:
                print(f"PGD Iteration {i}, Confidence: {confidence.item():.4f}")
                
        return patch.detach()
    
    def _simulate_detection_confidence(self, patch: torch.Tensor) -> torch.Tensor:
        """
        Simulate detection confidence for demonstration
        In practice, this would use a real detection model
        """
        # Simple heuristic based on patch characteristics
        mean_intensity = torch.mean(patch)
        variance = torch.var(patch)
        edge_strength = self._calculate_edge_strength(patch)
        
        # Higher variance and edge strength typically increase detectability
        confidence = 0.5 + 0.3 * variance + 0.2 * edge_strength
        return torch.clamp(confidence, 0, 1)
    
    def _calculate_edge_strength(self, patch: torch.Tensor) -> torch.Tensor:
        """Calculate edge strength in the patch"""
        # Convert to grayscale
        gray = 0.299 * patch[0] + 0.587 * patch[1] + 0.114 * patch[2]
        
        # Simple edge detection using differences
        dx = torch.abs(gray[1:, :] - gray[:-1, :])
        dy = torch.abs(gray[:, 1:] - gray[:, :-1])
        
        edge_strength = torch.mean(dx) + torch.mean(dy)
        return edge_strength
    
    def apply_color_constraints(self, patch: torch.Tensor, color_mode: str) -> torch.Tensor:
        """Apply color constraints based on the selected mode"""
        if color_mode == 'full_color':
            return patch
        elif color_mode == '4_color':
            # Quantize to 4 colors per channel
            patch = torch.round(patch * 3) / 3
        elif color_mode == 'black_white':
            # Convert to grayscale and threshold
            gray = 0.299 * patch[0] + 0.587 * patch[1] + 0.114 * patch[2]
            binary = (gray > 0.5).float()
            patch = torch.stack([binary, binary, binary])
        
        return torch.clamp(patch, 0, 1)
    
    def add_qr_overlay(self, patch: torch.Tensor, qr_data: str, 
                      position: str = 'center', size_ratio: float = 0.3) -> torch.Tensor:
        """Add QR code overlay to the patch"""
        # Convert patch to PIL Image
        patch_np = patch.permute(1, 2, 0).cpu().numpy()
        patch_pil = Image.fromarray((patch_np * 255).astype(np.uint8))
        
        # Generate QR code
        qr = qrcode.QRCode(version=1, box_size=10, border=1)
        qr.add_data(qr_data)
        qr.make(fit=True)
        qr_img = qr.make_image(fill_color="black", back_color="white")
        
        # Resize QR code
        qr_size = int(min(self.patch_size) * size_ratio)
        qr_img = qr_img.resize((qr_size, qr_size))
        
        # Calculate position
        if position == 'center':
            x = (self.patch_size[1] - qr_size) // 2
            y = (self.patch_size[0] - qr_size) // 2
        elif position == 'top_left':
            x, y = 10, 10
        elif position == 'top_right':
            x = self.patch_size[1] - qr_size - 10
            y = 10
        elif position == 'bottom_left':
            x = 10
            y = self.patch_size[0] - qr_size - 10
        elif position == 'bottom_right':
            x = self.patch_size[1] - qr_size - 10
            y = self.patch_size[0] - qr_size - 10
        else:
            x, y = (self.patch_size[1] - qr_size) // 2, (self.patch_size[0] - qr_size) // 2
        
        # Paste QR code with transparency
        patch_pil.paste(qr_img, (x, y))
        
        # Convert back to tensor
        patch_array = np.array(patch_pil) / 255.0
        return torch.from_numpy(patch_array).permute(2, 0, 1).float()
    
    def generate_patch(self, method: str = 'pgd', color_mode: str = 'full_color',
                      qr_data: Optional[str] = None, qr_position: str = 'center') -> torch.Tensor:
        """Generate adversarial patch with specified parameters"""
        print(f"Generating adversarial patch using {method} method...")
        
        # Initialize patch
        patch = self.initialize_patch('random')
        
        # Apply adversarial optimization
        if method == 'fgsm':
            patch = self.fgsm_attack(patch)
        elif method == 'pgd':
            patch = self.pgd_attack(patch)
        elif method == 'random':
            # Add some structured noise patterns
            patch = self._add_structured_patterns(patch)
        
        # Apply color constraints
        patch = self.apply_color_constraints(patch, color_mode)
        
        # Add QR code overlay if requested
        if qr_data:
            patch = self.add_qr_overlay(patch, qr_data, qr_position)
        
        self.patch = patch
        return patch
    
    def _add_structured_patterns(self, patch: torch.Tensor) -> torch.Tensor:
        """Add structured patterns that can confuse detection systems"""
        h, w = self.patch_size
        
        # Add geometric patterns
        for i in range(5):
            # Random circles
            center_x = random.randint(20, w-20)
            center_y = random.randint(20, h-20)
            radius = random.randint(5, 15)
            
            y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            
            color = torch.rand(3) * 0.5 + 0.25
            for c in range(3):
                patch[c][mask] = color[c]
        
        # Add some random lines
        for i in range(3):
            start_x = random.randint(0, w)
            start_y = random.randint(0, h)
            end_x = random.randint(0, w)
            end_y = random.randint(0, h)
            
            # Simple line drawing
            steps = max(abs(end_x - start_x), abs(end_y - start_y))
            if steps > 0:
                x_step = (end_x - start_x) / steps
                y_step = (end_y - start_y) / steps
                
                color = torch.rand(3) * 0.3 + 0.35
                for step in range(steps):
                    x = int(start_x + step * x_step)
                    y = int(start_y + step * y_step)
                    if 0 <= x < w and 0 <= y < h:
                        for c in range(3):
                            patch[c][y][x] = color[c]
        
        return patch
    
    def save_patch(self, filename: str, patch: Optional[torch.Tensor] = None):
        """Save the generated patch to file"""
        if patch is None:
            patch = self.patch
        
        if patch is None:
            raise ValueError("No patch to save. Generate a patch first.")
        
        # Convert to numpy array
        patch_np = patch.permute(1, 2, 0).cpu().numpy()
        patch_np = (patch_np * 255).astype(np.uint8)
        
        # Save using PIL
        patch_pil = Image.fromarray(patch_np)
        patch_pil.save(filename)
        print(f"Patch saved to {filename}")
    
    def visualize_patch(self, patch: Optional[torch.Tensor] = None, title: str = "Adversarial Patch"):
        """Visualize the generated patch"""
        if patch is None:
            patch = self.patch
            
        if patch is None:
            raise ValueError("No patch to visualize. Generate a patch first.")
        
        patch_np = patch.permute(1, 2, 0).cpu().numpy()
        
        plt.figure(figsize=(8, 8))
        plt.imshow(patch_np)
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Generate adversarial patches for AI camera evasion')
    parser.add_argument('--method', choices=['fgsm', 'pgd', 'random'], default='pgd',
                       help='Method for generating adversarial patch')
    parser.add_argument('--color-mode', choices=['full_color', '4_color', 'black_white'], 
                       default='full_color', help='Color mode for the patch')
    parser.add_argument('--size', type=int, nargs=2, default=[224, 224],
                       help='Size of the patch (height width)')
    parser.add_argument('--output', type=str, default='adversarial_patch.png',
                       help='Output filename for the patch')
    parser.add_argument('--qr-data', type=str, help='Data to encode in QR code overlay')
    parser.add_argument('--qr-position', choices=['center', 'top_left', 'top_right', 
                       'bottom_left', 'bottom_right'], default='center',
                       help='Position of QR code overlay')
    parser.add_argument('--visualize', action='store_true', 
                       help='Display the generated patch')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu',
                       help='Device to use for computation')
    
    args = parser.parse_args()
    
    print("=== Adversarial Patch Generator ===")
    print(f"Method: {args.method}")
    print(f"Color mode: {args.color_mode}")
    print(f"Patch size: {args.size[0]}x{args.size[1]}")
    print(f"Device: {args.device}")
    
    # Create generator
    generator = AdversarialPatchGenerator(
        patch_size=(args.size[0], args.size[1]),
        device=args.device
    )
    
    # Generate patch
    patch = generator.generate_patch(
        method=args.method,
        color_mode=args.color_mode,
        qr_data=args.qr_data,
        qr_position=args.qr_position
    )
    
    # Save patch
    generator.save_patch(args.output, patch)
    
    # Visualize if requested
    if args.visualize:
        generator.visualize_patch(patch, f"Adversarial Patch ({args.method}, {args.color_mode})")
    
    print(f"\nAdversarial patch generated successfully!")
    print(f"Output saved to: {args.output}")
    
    # Print usage suggestions
    print("\n=== Usage Suggestions ===")
    print("- Print the patch on high-quality paper or fabric")
    print("- Test different sizes and positions for optimal effectiveness")
    print("- Consider environmental factors (lighting, angle, distance)")
    print("- Combine with other evasion techniques for better results")
    
    if args.qr_data:
        print(f"- QR code contains: {args.qr_data}")
        print("- QR code can be used for additional functionality or misdirection")


if __name__ == "__main__":
    main()
