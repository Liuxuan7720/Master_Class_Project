import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import math


class AdaptiveImportanceController:
    
    def __init__(self, 
                 base_threshold: float = 0.1,
                 importance_gamma: float = 0.1,
                 min_opacity: float = 0.05,
                 max_pruning_ratio: float = 0.6,
                 viewpoint_sensitivity_window: int = 10):
        self.base_threshold = base_threshold
        self.importance_gamma = importance_gamma
        self.min_opacity = min_opacity
        self.max_pruning_ratio = max_pruning_ratio
        self.viewpoint_sensitivity_window = viewpoint_sensitivity_window 
        
        self.viewpoint_loss_history = []
        self.gaussian_contribution_history = []
    def clear_state(self):
        self.viewpoint_gradients_history = {} 
        self.viewpoint_loss_history = []
        self.gaussian_contribution_history = []
    def calculate_importance_score(self, 
                                 gaussians, 
                                 viewpoint_gradients: Dict[str, torch.Tensor],
                                 iteration: int) -> torch.Tensor:
        device = gaussians.get_xyz.device
        n_gaussians = gaussians.get_xyz.shape[0]
        
        opacity_weights = torch.sigmoid(gaussians.get_opacity).squeeze(-1)
        
        viewpoint_sensitivity = self._calculate_viewpoint_sensitivity(
            gaussians, viewpoint_gradients
        )
        
        density_compensation = self._calculate_density_compensation(gaussians)
        
        importance_scores = (opacity_weights * viewpoint_sensitivity * density_compensation)
        
        return importance_scores
    
    def _calculate_viewpoint_sensitivity(self, 
                                       gaussians, 
                                       viewpoint_gradients: Dict[str, torch.Tensor]) -> torch.Tensor:
        device = gaussians.get_xyz.device
        n_gaussians = gaussians.get_xyz.shape[0]
        
        if len(viewpoint_gradients) < 2:
            return torch.ones(n_gaussians, device=device)
        
        position_gradients = []
        for viewpoint_key, grads in viewpoint_gradients.items():
            if 'xyz' in grads:
                position_gradients.append(grads['xyz'])
        
        if len(position_gradients) < 2:
            return torch.ones(n_gaussians, device=device)
        
        grad_stack = torch.stack(position_gradients, dim=0)  # [n_views, n_gaussians, 3]
        grad_variance = torch.var(grad_stack, dim=0).mean(dim=-1)  # [n_gaussians]
        
        if grad_variance.max() > 0:
            grad_variance = grad_variance / grad_variance.max()
        
        return grad_variance
    
    def _calculate_density_compensation(self, gaussians) -> torch.Tensor:

        device = gaussians.get_xyz.device
        positions = gaussians.get_xyz  # [n_gaussians, 3]
        n_gaussians = positions.shape[0]
        
        K = min(50, n_gaussians // 10)
        
        batch_size = min(10000, n_gaussians)  
        knn_distances = torch.zeros(n_gaussians, K, device=device)
        
        for i in range(0, n_gaussians, batch_size):
            batch_positions = positions[i:i+batch_size]
            batch_distances = torch.cdist(batch_positions, positions)  # [batch_size, n_gaussians]
            batch_knn_distances, _ = torch.topk(batch_distances, k=K+1, largest=False, dim=-1)
            knn_distances[i:i+batch_size] = batch_knn_distances[:, 1:] 
        
        local_density = 1.0 / (knn_distances.mean(dim=-1) + 1e-8)
        

        if local_density.max() > local_density.min():
            local_density = (local_density - local_density.min()) / (local_density.max() - local_density.min())
        
        return local_density
    
    def calculate_dynamic_threshold(self, 
                                  gaussians, 
                                  scene_complexity_factor: float,
                                  iteration: int) -> float:
        dynamic_threshold = self.base_threshold * (1 + self.importance_gamma * scene_complexity_factor)
        
        progress_factor = min(1.0, iteration / 15000.0)
        dynamic_threshold *= (0.5 + 0.5 * progress_factor)
        
        return dynamic_threshold
    
    def calculate_scene_complexity(self, gaussians, iteration: int) -> float:
        current_n_gaussians = gaussians.get_xyz.shape[0]
        
        if len(self.gaussian_contribution_history) > 10:
            recent_counts = self.gaussian_contribution_history[-10:]
            growth_rate = (current_n_gaussians - recent_counts[0]) / len(recent_counts)
            complexity_factor = min(1.0, growth_rate / 1000.0)  # 标准化
        else:
            complexity_factor = 0.5  # 默认中等复杂度
        
        self.gaussian_contribution_history.append(current_n_gaussians)
        
        return complexity_factor
    
    def identify_pruning_candidates(self, 
                                  gaussians,
                                  importance_scores: torch.Tensor,
                                  dynamic_threshold: float,
                                  viewpoint_cameras: List) -> torch.Tensor:

        device = gaussians.get_xyz.device
        n_gaussians = gaussians.get_xyz.shape[0]
        
        low_importance_mask = importance_scores < dynamic_threshold
        
        opacity_values = torch.sigmoid(gaussians.get_opacity).squeeze(-1)
        low_opacity_mask = opacity_values < self.min_opacity
        
        scales = gaussians.get_scaling
        max_scale = scales.max(dim=-1)[0]
        large_scale_mask = max_scale > 0.1
        
        candidate_mask = low_importance_mask | (low_opacity_mask & large_scale_mask)
        

        n_candidates = candidate_mask.sum().item()
        max_pruning_count = int(n_gaussians * self.max_pruning_ratio)
        
        if n_candidates > max_pruning_count:
            candidate_indices = torch.where(candidate_mask)[0]
            candidate_scores = importance_scores[candidate_indices]
            _, sorted_indices = torch.sort(candidate_scores)
            
            final_pruning_mask = torch.zeros(n_gaussians, dtype=torch.bool, device=device)
            final_pruning_indices = candidate_indices[sorted_indices[:max_pruning_count]]
            final_pruning_mask[final_pruning_indices] = True
            
            return final_pruning_mask
        
        return candidate_mask
    
    def verify_pruning_candidates(self, 
                                gaussians,
                                pruning_mask: torch.Tensor,
                                viewpoint_cameras: List,
                                renderer) -> torch.Tensor:
        device = gaussians.get_xyz.device
        
        positions = gaussians.get_xyz
        pruning_positions = positions[pruning_mask]
        remaining_positions = positions[~pruning_mask]
        
        if remaining_positions.shape[0] == 0:
            return torch.zeros_like(pruning_mask)
        
        distances = torch.cdist(pruning_positions, remaining_positions)
        min_distances, _ = torch.min(distances, dim=-1)
        
        coverage_threshold = 0.05 
        well_covered_mask = min_distances < coverage_threshold
        
        verified_mask = torch.zeros_like(pruning_mask)
        verified_mask[torch.where(pruning_mask)[0][well_covered_mask]] = True
        
        return verified_mask
    
    def progressive_pruning(self, 
                          gaussians,
                          pruning_mask: torch.Tensor,
                          current_iteration: int,
                          pruning_interval: int = 100) -> torch.Tensor:
        device = gaussians.get_xyz.device
        n_to_prune = pruning_mask.sum().item()
        
        if n_to_prune == 0:
            return pruning_mask
        
        pruning_phase = (current_iteration // pruning_interval) % 3 
        
        if pruning_phase == 0:
            actual_pruning_count = int(n_to_prune * 0.3)
        elif pruning_phase == 1:
            actual_pruning_count = int(n_to_prune * 0.5)
        else:
            actual_pruning_count = int(n_to_prune * 0.2)
        
        if actual_pruning_count == 0:
            return torch.zeros_like(pruning_mask)
        
        pruning_indices = torch.where(pruning_mask)[0]
        if len(pruning_indices) > actual_pruning_count:
            selected_indices = pruning_indices[torch.randperm(len(pruning_indices))[:actual_pruning_count]]
            progressive_mask = torch.zeros_like(pruning_mask)
            progressive_mask[selected_indices] = True
            return progressive_mask
        
        return pruning_mask
    
    def update_viewpoint_gradients(self, 
                                 viewpoint_key: str,
                                 gradients: Dict[str, torch.Tensor]):
        if not hasattr(self, 'viewpoint_gradients_history'):
            self.viewpoint_gradients_history = {}
        
        if viewpoint_key not in self.viewpoint_gradients_history:
            self.viewpoint_gradients_history[viewpoint_key] = []
        
        self.viewpoint_gradients_history[viewpoint_key].append(gradients)
        
        if len(self.viewpoint_gradients_history[viewpoint_key]) > self.viewpoint_sensitivity_window:
            self.viewpoint_gradients_history[viewpoint_key].pop(0)
    
    def get_current_viewpoint_gradients(self) -> Dict[str, torch.Tensor]:

        if not hasattr(self, 'viewpoint_gradients_history'):
            return {}
        
        current_gradients = {}
        for viewpoint_key, history in self.viewpoint_gradients_history.items():
            if history:
                current_gradients[viewpoint_key] = history[-1]
        
        return current_gradients