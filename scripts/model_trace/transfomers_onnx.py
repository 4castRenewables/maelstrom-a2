# import onnx
# onnx_model = onnx.load("ap2.onnx")
# poetry run python -m transformers.onnx --model=microsoft/deberta-v3-small  onnx/ --atol 3e-5
from onnxruntime import InferenceSession
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")

session = InferenceSession("onnx/model.onnx")

# ONNX Runtime expects NumPy arrays as input

inputs = tokenizer("Using DistilBERT with ONNX Runtime!", return_tensors="np")
inputs.pop("token_type_ids")
outputs = session.run(output_names=["logits"], input_feed=dict(inputs))
