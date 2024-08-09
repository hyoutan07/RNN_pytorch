import torch
print(torch.__version__)  # PyTorchのバージョンを確認
print(torch.backends.mps.is_available())  # MPSが利用可能か確認
print(torch.backends.mps.is_built())  # MPSがPyTorchにビルドされているか確認

