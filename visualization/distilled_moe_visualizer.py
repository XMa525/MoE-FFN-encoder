import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import matplotlib.colors as mcolors

class MoEExpertVisualizer:
    def __init__(self, num_experts=5, shared_expert_id=4):
        """
        num_experts: total experts (including shared)
        shared_expert_id: shared expert ID to ignore
        """
        self.num_experts = num_experts
        self.shared_expert_id = shared_expert_id
        # 颜色映射
        self.colors = ["red", "blue", "green", "purple", "orange", "cyan", "magenta", "yellow"][:num_experts]
    # Utility: detect WSI
    # -------------------------------
    def is_wsi(self, img_path):
        ext = os.path.splitext(img_path)[-1].lower()
        return ext in ['.tif', '.svs', '.ndpi']

    # -------------------------------
    # Load image
    # -------------------------------
    def load_image(self, img_path):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    # -------------------------------
    # Compute grid
    # -------------------------------
    def tokens_to_grid(self, N_tokens):
        grid_size = int(np.sqrt(N_tokens))
        return grid_size, grid_size

    # -------------------------------
    # Winner map: segmentation-style
    # -------------------------------
    def plot_winner_map(self, img, dispatch_weights):
        """
        dispatch_weights: [N_tokens, N_experts] (torch.Tensor)
        """
        dispatch = dispatch_weights.cpu().numpy()
        winner = np.argmax(dispatch, axis=1)  # 每个 token 主导专家
        winner[winner == self.shared_expert_id] = -1  # shared expert 用黑色表示

        grid_h, grid_w = self.tokens_to_grid(dispatch.shape[0])
        seg_map = winner.reshape(grid_h, grid_w)

        plt.figure(figsize=(6,6))
        plt.imshow(img)
        cmap = plt.cm.get_cmap("tab10", self.num_experts)
        overlay = plt.imshow(seg_map, cmap=cmap, alpha=0.5, vmin=-1, vmax=self.num_experts-1)
        plt.title("Winner Map (Segmentation-style)")
        plt.axis('off')
        plt.show()

    # -------------------------------
    # Scatter map
    # -------------------------------
    def plot_scatter_map(self, img, dispatch_weights):
        dispatch = dispatch_weights.cpu().numpy()
        winner = np.argmax(dispatch, axis=1)
        winner[winner == self.shared_expert_id] = -1  # 排除 shared

        N_tokens = dispatch.shape[0]
        grid_h, grid_w = self.tokens_to_grid(N_tokens)
        h, w, _ = img.shape
        patch_h = h // grid_h
        patch_w = w // grid_w

        fig, ax = plt.subplots(figsize=(6,6))
        ax.imshow(img)

        for token_id in range(N_tokens):
            e = winner[token_id]
            if e == -1:
                continue
            y = token_id // grid_w
            x = token_id % grid_w
            cx = x * patch_w + patch_w // 2 + np.random.uniform(-1,1)
            cy = y * patch_h + patch_h // 2 + np.random.uniform(-1,1)
            prob = dispatch[token_id, e]
            ax.scatter(cx, cy, s=5 + prob*30, c=self.colors[e], alpha=0.7, marker='s')

        ax.axis('off')
        plt.title("Expert Scatter Map")
        plt.show()

    # -------------------------------
    # Expert activation heatmap
    # -------------------------------
    def plot_activation_heatmaps(self, img, dispatch_weights):
        dispatch = dispatch_weights.cpu().numpy()
        N_tokens, N_experts = dispatch.shape
        grid_h, grid_w = self.tokens_to_grid(N_tokens)
        h, w, _ = img.shape
        patch_h = h // grid_h
        patch_w = w // grid_w

        for e in range(N_experts):
            if e == self.shared_expert_id:
                continue
            heat = np.zeros((grid_h, grid_w))
            for token_id in range(N_tokens):
                y = token_id // grid_w
                x = token_id % grid_w
                heat[y, x] = dispatch[token_id, e]

            heat_resized = cv2.resize(heat, (w, h), interpolation=cv2.INTER_CUBIC)
            heat_resized = (heat_resized - heat_resized.min()) / (heat_resized.max()-heat_resized.min()+1e-6)
            heatmap = cv2.applyColorMap((heat_resized*255).astype(np.uint8), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

            plt.figure(figsize=(6,6))
            plt.imshow(overlay)
            plt.title(f"Expert {e} Activation Heatmap")
            plt.axis('off')
            plt.show()

    # WSI visualization
    # -------------------------------
    def visualize_wsi(self, wsi_path, dispatch_weights, coords, patch_size=256):
        import openslide
        wsi = openslide.OpenSlide(wsi_path)
        w, h = wsi.dimensions
        canvas = np.array(wsi.read_region((0,0),0,(w,h)))[:,:,:3]

        winner = np.argmax(dispatch_weights.cpu().numpy(), axis=1)
        if self.shared_expert_id is not None:
            winner[winner==self.shared_expert_id] = -1


        for i, (x, y) in enumerate(coords):
            e = winner[i]
            if e == -1:
                continue
            color = np.array(mcolors.to_rgb(self.colors[e]))  # 转为 0~1 RGB
            canvas[y:y+patch_size, x:x+patch_size,:] = (
                0.6*canvas[y:y+patch_size, x:x+patch_size,:] + 0.4*color*255
            ).astype(np.uint8)

        plt.figure(figsize=(12,12))
        plt.imshow(canvas)
        plt.axis('off')
        plt.show()

    # -------------------------------
    # Main visualize
    # -------------------------------
    def visualize(self, img_path, dispatch_weights):
        """
        dispatch_weights: [1, N_tokens, N_experts] or [N_tokens, N_experts]
        """
        img = self.load_image(img_path)
        if len(dispatch_weights.shape) == 3:
            dispatch_weights = dispatch_weights.squeeze(0)
        # Winner map
        self.plot_winner_map(img, dispatch_weights)
        # Scatter map
        self.plot_scatter_map(img, dispatch_weights)
        # Expert activation heatmaps
        self.plot_activation_heatmaps(img, dispatch_weights)