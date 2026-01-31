import torch
import numpy as np
import torch.nn as nn

import onnx, onnxruntime
from onnxsim import simplify

valid = torch.tensor(np.load("./output/index/valid.npy")).cuda()
x = torch.tensor(np.load("./output/index/x.npy")).cuda()
y = torch.tensor(np.load("./output/index/y.npy")).cuda()
volume = torch.tensor(np.load("./output/index/volume.npy")).cuda()
img = torch.tensor(np.load("./output/index/features.npy")).cuda()
features = img  # 6,64,64,176

gather_index_onnx = []
scather_index_onnx = []

out1 = torch.zeros(64, 160000).cuda()

for i in range(6):
    feature = torch.flatten(features[i].permute(1, 2, 0), 0, 1)  # 64,64,176 -> 64,176,64 -> 11264,64
    y_index = y[i, valid[i]]  # 0-63
    x_index = x[i, valid[i]]  # 0-175
    gather_index = (y_index * 176 + x_index)

    scather_index = torch.arange(160000)[valid[i]].cuda()

    update = feature[gather_index, :].permute(1, 0)

    out1[:, scather_index] = update
    
    gather_index_onnx.append(gather_index)
    scather_index_onnx.append(scather_index)

for i in range(6):
    volume[:, valid[i]] = features[i, :, y[i, valid[i]], x[i, valid[i]]]

class Toymodel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, features):
        out2 = torch.zeros(64, 160000)
        for i in range(6):

            feature = torch.flatten(features[i].cpu().permute(1, 2, 0), start_dim=0, end_dim=1)  # 64,64,176 -> 64,176,64 -> 11264,64

            update = feature[gather_index_onnx[i].cpu(), :].permute(1, 0)
    
            out2[:, scather_index_onnx[i].cpu()] = update

        return out2.reshape(1, 64, 200, 200, 4).permute(0, 4, 1, 2, 3).reshape(1, 256, 200, 200)

Net = Toymodel()
torch.onnx.export(Net, features, './2d_to_3d.onnx', opset_version=13)
onnx_model_2d = onnx.load('./2d_to_3d.onnx')
onnx.checker.check_model(onnx_model_2d)
simplified_model_2d, check_ok = simplify(onnx_model_2d)
onnx.save_model(simplified_model_2d, f'./simplified_export_2d_to_3d.onnx')