import torch
import random

def generate_block_mask(batch_size, grid_size=16, mask_ratio=0.3, block_size=3):
    """
    生成 Block Mask
    针对 224x224 图像，patch_size=14，grid_size = 224//14 = 16
    如果图像/Patch尺寸不同，调用时修改 grid_size
    """
    N = grid_size * grid_size
    num_masking_patches = int(N * mask_ratio)
    
    mask = torch.zeros((batch_size, grid_size, grid_size), dtype=torch.bool)
    
    for b in range(batch_size):
        masked_count = 0
        while masked_count < num_masking_patches:
            h_start = random.randint(0, grid_size - block_size)
            w_start = random.randint(0, grid_size - block_size)
            
            mask[b, h_start:h_start+block_size, w_start:w_start+block_size] = True
            masked_count = mask[b].sum().item()
            
            if masked_count >= num_masking_patches:
                break
                
    return mask.view(batch_size, N)