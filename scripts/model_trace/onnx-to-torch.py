import onnx
import torch
from onnx2torch import convert

# Path to ONNX model
onnx_model_path = "onnx/model.onnx"

# Or you can load a regular onnx model and pass it to the converter
onnx_model = onnx.load(onnx_model_path)
torch_model_2 = convert(onnx_model)
torch.save(torch_model_2, "ap2.pt")
