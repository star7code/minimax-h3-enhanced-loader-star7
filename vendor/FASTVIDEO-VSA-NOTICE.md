# FastVideo VSA Triton kernels

`vsa_triton/block_sparse_attn_triton.py` and `vsa_triton/index.py` are adapted
from the Apache-2.0 licensed
[hao-ai-lab/FastVideo](https://github.com/hao-ai-lab/FastVideo) project,
`fastvideo-kernel/python/fastvideo_kernel/triton_kernels/`.

Only the CUDA-agnostic Triton inference path required by MiniMax H3 tile-64
VSA is bundled. FastVideo's optional ThunderKittens, CuTe/FA4, sm100a and
training/autograd wrappers are not included.
