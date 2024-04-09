import torch
import torch.onnx

import sys
sys.path.append('/home/shiroha/Code/pytorch/Lenet') # 修改model的路径
from model import LeNet                             # 修改model的名字

import os

def pth_to_onnx(pth_path, input, onnx_path, input_names=['input'], output_names=['output'], device='cpu'):
    if not onnx_path.endswith('.onnx'):
        print('Warning! The onnx model name is not correct,\
              please give a name that ends with \'.onnx\'!')
        return 0

    model = torch.load(pth_path) #LeNet() #导入模型
    # model.load_state_dict(torch.load(pth_path)) #初始化权重
    # model.eval()
    
    torch.onnx.export(model, input, onnx_path, verbose=True, input_names=input_names, output_names=output_names) #指定模型的输入，以及onnx的输出路径
    print("Exporting .pth model to onnx model has been successful!")

if __name__ == '__main__':
    pth_path = '/home/shiroha/Code/pytorch/Lenet/models/pth/lenet-model.pth' # 修改input的pth
    onnx_path = '../Lenet/models/onnx/lenet.onnx'                            # 修改output的onnx
    input = torch.randn(1, 1, 28, 28)
    pth_to_onnx(pth_path, input,  onnx_path)
