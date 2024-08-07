import torch

# サイズ [10, 100] のテンソルを作成
tensor1 = torch.randn(10, 5, 100)

# サイズ [4, 50] のテンソルを作成
tensor2 = torch.randn(10, 100)

# 代入によるエラーの発生
tensor1[:, 4, :] = tensor2
