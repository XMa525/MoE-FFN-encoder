import torch
import numpy as np
import os

class EarlyStopping:
    """早停工具类：监控验证集 Loss，当连续 patience 个 epoch 没下降时停止训练"""
    def __init__(self, patience=5, min_delta=1e-4, save_path="moe_encoder_distilled.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

        # 平台期提前停
        # self.plateau_patience = plateau_patience
        # self.eps = eps
        # self.plateau_counter = 0
        # self.last_loss = None

    def __call__(self, val_loss, model):
        # 如果当前的 val_loss 比历史最佳的还要好（且超出了最小改善阈值）
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(val_loss, model)
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience} (Best: {self.best_loss:.6f})")
            if self.counter >= self.patience:
                self.early_stop = True
                return
        # # =========================
        # # 平台期提前停
        # # =========================
        # if self.last_loss is not None:
        #     if abs(val_loss - self.last_loss) < self.eps:
        #         self.plateau_counter += 1
        #         print(f"Plateau counter: {self.plateau_counter} out of {self.plateau_patience} (delta < {self.eps})")
        #         if self.plateau_counter >= self.plateau_patience:
        #             print("⚠️ Plateau early stop triggered. Training loss change very small.")
        #             self.early_stop = True
        #             return
        #     else:
        #         self.plateau_counter = 0

        # self.last_loss = val_loss


    def save_checkpoint(self, val_loss, model):
        """当验证集 Loss 下降时保存模型"""
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        print(f"🌟 Validation loss decreased to {val_loss:.6f}. Saving best model...")
        # 注意：这里我们只保存学生模型（DINOv2 MoE）的权重
        torch.save(model.state_dict(), self.save_path)
        print("✅ Model saved.")