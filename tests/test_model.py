import torch
from codellm.model import RMSNorm
from transformers.models.llama.modeling_llama import LlamaRMSNorm


def test_rmsnorm():
    batch_size, seq_len, dim = 2, 5, 16

    hf_rmsnorm = LlamaRMSNorm(hidden_size=dim)
    rmsnorm = RMSNorm(dim=dim)
    rmsnorm.gamma.data.copy_(hf_rmsnorm.weight.data)

    x = torch.randn(batch_size, seq_len, dim) * 0.02
    output_hf = hf_rmsnorm(x)
    output_custom = rmsnorm(x)

    x_zeros = torch.zeros(batch_size, seq_len, dim)
    output_hf_zeros = hf_rmsnorm(x_zeros)
    output_custom_zeros = rmsnorm(x_zeros)

    torch.testing.assert_close(output_custom, output_hf)
    torch.testing.assert_close(output_custom_zeros, output_hf_zeros)
