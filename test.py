from diffusers import AutoPipelineForInpainting

import torch


pipeline_name = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
pipeline =  AutoPipelineForInpainting.from_pretrained(
    pipeline_name,
    torch_dtype=torch.float16
)# YOUR CODE HERE

# This will make it more efficient on our hardware
pipeline.enable_model_cpu_offload()