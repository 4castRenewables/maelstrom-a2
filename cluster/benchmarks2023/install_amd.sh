wget https://download.pytorch.org/whl/nightly/rocm5.2/torch-2.0.0.dev20230209%2Brocm5.2-cp310-cp310-linux_x86_64.whl
wget https://download.pytorch.org/whl/nightly/rocm5.2/torchvision-0.15.0.dev20230210%2Brocm5.2-cp310-cp310-linux_x86_64.whl
pip install a2 datasets transformers sentence-transformers sentencepiece
pip uninstall torch torchvision
pip install torch-2.0.0.dev20230209+rocm5.2-cp310-cp310-linux_x86_64.whl
pip install torchvision-0.15.0.dev20230210+rocm5.2-cp310-cp310-linux_x86_64.whl