import matplotlib.pyplot as plt
import numpy as np
import os


class DistillVisualizer:

    def __init__(self):
        self.train_history = {
            "total_loss": [],
            "cls_l12": [],
            "patch_masked_l12": [],
            "patch_unmasked_l12": [],
            "moe_loss": [],
            "prc_loss": [],
            "entropy": [],
        }

        self.val_history = {
            "total_loss": [],
            "cls_l12": [],
            "patch_masked_l12": [],
            "patch_unmasked_l12": [],
            "moe_loss": [],
            "prc_loss": [],
            "entropy": [],
        }

        self.train_expert_usage = []
        self.val_expert_usage = []

        
    def update(self, loss_dict, entropy=None, expert_usage=None, mode="train"):
        history = self.train_history if mode == "train" else self.val_history
        usage = self.train_expert_usage if mode == "train" else self.val_expert_usage
        for k in history.keys():
            if k in loss_dict:
                history[k].append(loss_dict[k])

        if entropy is not None:
            history["entropy"].append(entropy)

        if expert_usage is not None:
            usage.append(expert_usage)
    def smooth(self, x, w=20):
        if len(x) < w:
            return x
        return np.convolve(x, np.ones(w)/w, mode="valid")

    # ---------------- Loss 曲线 ----------------
    def plot_loss(self, mode="train",save_path=None):
        history = self.train_history if mode == "train" else self.val_history
        steps = np.arange(len(history["total_loss"]))

        plt.figure(figsize=(10,6))

        #平滑曲线
        plt.plot(np.arange(len(self.smooth(history["total_loss"]))), 
         self.smooth(history["total_loss"]), label="Total Loss", linewidth=2)

        plt.plot(np.arange(len(self.smooth(history["cls_l12"]))), 
                self.smooth(history["cls_l12"]), label="CLS Loss")
        plt.plot(np.arange(len(self.smooth(history["patch_masked_l12"]))), 
                self.smooth(history["patch_masked_l12"]), label="Masked Patch Loss")
        plt.plot(np.arange(len(self.smooth(history["patch_unmasked_l12"]))), 
                self.smooth(history["patch_unmasked_l12"]), label="Unmasked Patch Loss")
        plt.plot(np.arange(len(self.smooth(history["moe_loss"]))), 
                self.smooth(history["moe_loss"]), label="MoE Loss")
        plt.plot(np.arange(len(self.smooth(history["prc_loss"]))), 
                self.smooth(history["prc_loss"]), label="PRC Loss")

        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.title("Distillation Training Convergence")

        plt.legend()
        plt.grid(alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300)

        #plt.show()
        plt.close()

    # ---------------- Entropy 曲线 ----------------
    def plot_entropy(self, mode="train" ,save_path=None):
        history = self.train_history if mode == "train" else self.val_history
        usage = self.train_expert_usage if mode == "train" else self.val_expert_usage

        if len(history["entropy"]) == 0 or len(usage) == 0:
            return

        steps = np.arange(len(history["entropy"]))

        usage_array = np.stack(usage)
        num_experts = usage_array.shape[1] - 1
        max_entropy = np.log(num_experts)

        plt.figure(figsize=(8,5))

        plt.plot(steps, history["entropy"], linewidth=2, label="Routing Entropy")
        # MoE健康范围参考
        plt.axhline(0.5 * max_entropy, linestyle="--", alpha=0.5, label="Healthy Lower Bound")
        plt.axhline(0.9 * max_entropy, linestyle="--", alpha=0.5, label="Healthy Upper Bound")

        plt.xlabel("Training Step")
        plt.ylabel("Routing Entropy")

        plt.title("MoE Routing Entropy Evolution")
        plt.legend()
        plt.grid(alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300)

        #plt.show()
        plt.close()

    # ---------------- Expert Usage ----------------
    def plot_expert_usage(self, mode="train", save_path=None):
        usage = self.train_expert_usage if mode == "train" else self.val_expert_usage
        if len(usage) == 0:
            return

        usage_array = np.stack(usage)

        plt.figure(figsize=(10,6))

        for i in range(usage_array.shape[1]-1): # 最后一个是 shared expert
            plt.plot(usage_array[:, i], label=f"Expert {i}")

        

        plt.xlabel("Training Step")
        plt.ylabel("Usage Probability")

        plt.title("Expert Usage Evolution")
        plt.legend()
        plt.grid(alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300)

        #plt.show()
        plt.close()

    # ---------------- Expert Histogram（判断collapse） ----------------
    def plot_expert_histogram(self,mode="train", save_path=None):

        usage = self.train_expert_usage if mode == "train" else self.val_expert_usage

        if len(usage) == 0:
            return

        usage_array = np.stack(usage)

        # 使用整个训练平均
        mean_usage = usage_array[:, :-1].mean(axis=0)

        plt.figure(figsize=(8,5))

        x = np.arange(len(mean_usage))

        plt.bar(x, mean_usage)

        plt.xlabel("Expert ID")
        plt.ylabel("Average Usage")

        plt.title("Expert Usage Histogram (Collapse Check)")

        plt.grid(alpha=0.3, axis="y")

        if save_path:
            plt.savefig(save_path, dpi=300)

        #plt.show()
        plt.close()

    # ---------------- 总结 ----------------
    def summarize(self):

        os.makedirs("results/distill", exist_ok=True)

        for mode in ["train","val"]:
            self.plot_loss(mode, f"results/distill/{mode}_loss_curve.png")
            self.plot_entropy(mode, f"results/distill/{mode}_entropy_curve.png")
            self.plot_expert_usage(mode, f"results/distill/{mode}_expert_usage_curve.png")
            self.plot_expert_histogram(mode, f"results/distill/{mode}_expert_histogram.png")
        # self.plot_loss()
        # self.plot_entropy()
        # self.plot_expert_usage()
        # self.plot_expert_histogram()