from cs336_alignment.grpo import masked_mean
from cs336_alignment.sft_utils import masked_normalize
import torch

def test_masked_mean():
    ratio = torch.tensor([
        [1, 1, 1, 1, 1, 1, 1,],
        [1, 1, 1, 1, 1, 1, 1,],
    ], requires_grad=True)
    advs = torch.tensor([
        [2, 2, 2, 2, 2, 2, 2,],
        [2, 2, 2, 2, 2, 2, 2,],
    ])
    masks = torch.tensor([
    # generation 1: 4 tokens 
        [1, 1, 1, 1, 0, 0, 0,], 
    # # generation 2: 7 tokens 
        [1, 1, 1, 1, 1, 1, 1,],
    ])
    # Normalize with each approach
    masked_mean_result = masked_mean(ratio * advs, masks, dim=1)
    
    print("masked_mean", masked_mean_result)
    
    # masked_mean tensor([2., 2.], grad_fn=<DivBackward0>)
    
    masked_mean_result.mean().backward() 
    print("ratio.grad", ratio.grad)
    # ratio.grad:
    # tensor([[0.2500, 0.2500, 0.2500, 0.2500, 0.0000, 0.0000, 0.0000], # [0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429]]) ratio.grad.zero_()
    
def test_masked_normlized():
    ratio = torch.tensor([
        [1, 1, 1, 1, 1, 1, 1,],
        [1, 1, 1, 1, 1, 1, 1,],
    ], requires_grad=True)
    advs = torch.tensor([
        [2, 2, 2, 2, 2, 2, 2,],
        [2, 2, 2, 2, 2, 2, 2,],
    ])
    masks = torch.tensor([
    # generation 1: 4 tokens 
        [1, 1, 1, 1, 0, 0, 0,], 
    # # generation 2: 7 tokens 
        [1, 1, 1, 1, 1, 1, 1,],
    ])
    # Normalize with each approach
    max_gen_len = 7
    masked_normalize_result = masked_normalize(
        ratio * advs, masks, dim=1, constant_normalizer=max_gen_len)
    print("masked_normalize", masked_normalize_result)
    # masked_normalize tensor([1.1429, 2.0000], grad_fn=<DivBackward0>)
    masked_normalize_result.mean().backward()
    print("ratio.grad", ratio.grad)
    # ratio.grad:
    # tensor([[0.1429, 0.1429, 0.1429, 0.1429, 0.0000, 0.0000, 0.0000], # [0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429]])

if __name__ == "__main__":
    test_masked_mean()
    test_masked_normlized()