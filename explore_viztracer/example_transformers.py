from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "facebook/opt-125m"
model = AutoModelForCausalLM.from_pretrained(model_name)
model.cuda()

tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = ["DeepSpeed is a machine learning framework"]

input_tokens = tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding=True)
for t in input_tokens:
    if torch.is_tensor(input_tokens[t]):
        input_tokens[t] = input_tokens[t].to("cuda:0")
generate_kwargs = dict(max_new_tokens=100, do_sample=False)
outputs = model.generate(**input_tokens, **generate_kwargs)

outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(outputs)