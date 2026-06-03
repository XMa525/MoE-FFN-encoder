import torch

orig = torch.load("../UNI/pytorch_model.bin", map_location="cpu", weights_only=False)
if "model_state_dict" in orig:
    orig = orig["model_state_dict"]
elif "state_dict" in orig:
    orig = orig["state_dict"]
elif "encoder" in orig:
    orig = orig["encoder"]

new = torch.load("results/BRACS_uni_lora_v2/best_encoder_for_extract.pth", map_location="cpu", weights_only=False)
if "model_state_dict" in new:
    new = new["model_state_dict"]
elif "state_dict" in new:
    new = new["state_dict"]
elif "encoder" in new:
    new = new["encoder"]

def clean(sd):
    out = {}
    for k, v in sd.items():
        k = k.replace("model.", "")
        if k.startswith("module."):
            k = k[len("module."):]
        out[k] = v
    return out

orig = clean(orig)
new = clean(new)

common = sorted(set(orig.keys()) & set(new.keys()))
print("num common keys:", len(common))

total_abs = 0.0
total_num = 0

top_diffs = []
for k in common:
    if orig[k].shape != new[k].shape:
        continue
    diff = (orig[k].float() - new[k].float()).abs()
    mean_diff = diff.mean().item()
    max_diff = diff.max().item()
    top_diffs.append((mean_diff, max_diff, k))
    total_abs += diff.sum().item()
    total_num += diff.numel()

print("global mean abs diff:", total_abs / max(1, total_num))
top_diffs = sorted(top_diffs, reverse=True)[:20]
for mean_diff, max_diff, k in top_diffs:
    print(f"{k}: mean={mean_diff:.8f}, max={max_diff:.8f}")