import torch
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
