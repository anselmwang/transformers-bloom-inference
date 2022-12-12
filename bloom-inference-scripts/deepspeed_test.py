# deepspeed_test.py
import os

import deepspeed
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name = "facebook/opt-2.7b"

local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")

model = deepspeed.init_inference(
    model,
    mp_size=world_size,
    dtype=model.dtype,
    replace_method="auto",
    replace_with_kernel_inject=True,
)

test = "test prompt"
max_new_tokens = 100

generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer, device=local_rank)

# result = generator(test, do_sample=True, max_new_tokens=max_new_tokens)
result = generator(test, do_sample=True)

if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    print(result)