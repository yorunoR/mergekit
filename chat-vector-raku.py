from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

access_token=os.environ["HF_TOKEN"]

base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    token=access_token
)
chat_model = AutoModelForCausalLM.from_pretrained(
    "TFMC/Japanese-Starling-ChatV-7B",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)
cp_model = AutoModelForCausalLM.from_pretrained(
    "Rakuten/RakutenAI-7B",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)

tokenizer = AutoTokenizer.from_pretrained('Rakuten/RakutenAI-7B')

for k, v in cp_model.state_dict().items():
    chat_size = chat_model.state_dict()[k].shape[0]
    base_size = base_model.state_dict()[k].shape[0]
    cp_size = cp_model.state_dict()[k].shape[0]

    if chat_size == base_size:
        chat_vector = chat_model.state_dict()[k] - base_model.state_dict()[k]
    else:
        chat_vector = chat_model.state_dict()[k][:base_size, :] - base_model.state_dict()[k]

    if cp_size == base_size:
        new_v = v + ( 1.0 * chat_vector.to(v.device) )
    else:
        new_v = v

    v.copy_(new_v)

cp_model.save_pretrained("./1.0-chatvector-raku")
tokenizer.save_pretrained("./1.0-chatvector-raku")
