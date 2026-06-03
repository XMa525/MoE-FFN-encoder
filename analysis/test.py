import torch
obj = torch.load("features/parotid_virchow2_moe_feats/v7_adapt_424/22-36133A01_A02_程.pt", map_location="cpu", weights_only=False)
print(obj.keys())