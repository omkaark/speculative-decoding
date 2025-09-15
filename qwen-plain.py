import torch
from transformers import CLIPImageProcessor, AutoTokenizer
import torch
import json
from safetensors.torch import load_file 

 # RoPE
def rotate_half(t: torch.Tensor) -> torch.Tensor:
    d2 = t.shape[-1] // 2
    return torch.cat((-t[..., d2:], t[..., :d2]), dim=-1)

def apply_rope(qk: torch.Tensor, base: float) -> torch.Tensor:
    _, _, S, D = qk.shape
    assert D % 2 == 0
    inv = torch.arange(0, D, 2, device=qk.device, dtype=torch.float32) / D
    inv_freq = base ** (-inv) # (D/2,)
    t = torch.arange(S, device=qk.device, dtype=torch.float32) # (S,)
    freqs = torch.einsum("s,d->sd", t, inv_freq) # (S, D/2)
    emb = torch.cat([freqs, freqs], dim=-1) # (S, D)
    cos = emb.cos().to(qk.dtype)[None, None, :, :] # (1,1,S,D)
    sin = emb.sin().to(qk.dtype)[None, None, :, :]
    return (qk * cos) + (rotate_half(qk) * sin)

class Qwen2:
    def __init__(self, model_path):
        self.device = torch.device('mps')
        self.model = load_file(f"{model_path}/model.safetensors", device=self.device.type)

        # Tokenize and embed
        self.tokenizer = AutoTokenizer.from_pretrained(f"{model_path}")
        with open(f"{model_path}/config.json", "r") as f:
            self.config = json.load(f)
            
        self.embed = torch.nn.Embedding(self.config["vocab_size"], self.config["hidden_size"], device='mps', dtype=torch.bfloat16)
        self.embed.load_state_dict({"weight": self.model["model.embed_tokens.weight"]})

    def forward_decode_layers(self, x):
        S = x.shape[1]
        hidden = self.config["hidden_size"]
        n_heads = self.config["num_attention_heads"]
        n_kv = self.config["num_key_value_heads"]
        head_dim = hidden // n_heads
        for layer in range(self.config["num_hidden_layers"]):
            # RMSNorm
            x_rms = torch.nn.functional.rms_norm(
                x, normalized_shape=(hidden,),
                weight=self.model[f"model.layers.{layer}.input_layernorm.weight"],
                eps=self.config["rms_norm_eps"],
            ).to(torch.bfloat16)

            # QKV
            q = x_rms @ self.model[f"model.layers.{layer}.self_attn.q_proj.weight"].T + self.model[f"model.layers.{layer}.self_attn.q_proj.bias"]
            k = x_rms @ self.model[f"model.layers.{layer}.self_attn.k_proj.weight"].T + self.model[f"model.layers.{layer}.self_attn.k_proj.bias"]
            v = x_rms @ self.model[f"model.layers.{layer}.self_attn.v_proj.weight"].T + self.model[f"model.layers.{layer}.self_attn.v_proj.bias"]

            q = q.view(1, S, n_heads, head_dim).transpose(1, 2) # [1,H,S,D]
            k = k.view(1, S, n_kv,   head_dim).transpose(1, 2) # [1,KV,S,D]
            v = v.view(1, S, n_kv,   head_dim).transpose(1, 2)

            if n_heads != n_kv:
                reps = n_heads // n_kv
                k = k.repeat_interleave(reps, dim=1)
                v = v.repeat_interleave(reps, dim=1)

            # RoPE
            theta = self.config["rope_theta"]
            q = apply_rope(q, theta)
            k = apply_rope(k, theta)

            # Attn
            scores = (q @ k.transpose(-2, -1)) / (head_dim ** 0.5) # [1,H,S,S]
            causal = torch.triu(torch.full((S, S), float("-inf"), device='mps', dtype=scores.dtype), 1)
            attn = torch.softmax(scores + causal, dim=-1)
            attn_o = (attn @ v).transpose(1, 2).reshape(1, S, hidden) @ self.model[f"model.layers.{layer}.self_attn.o_proj.weight"].T

            x = x + attn_o

            # FFN
            x_rms = torch.nn.functional.rms_norm(
                x, normalized_shape=(hidden,),
                weight=self.model[f"model.layers.{layer}.post_attention_layernorm.weight"],
                eps=self.config["rms_norm_eps"],
            ).to(torch.bfloat16)

            gate = x_rms @ self.model[f"model.layers.{layer}.mlp.gate_proj.weight"].T
            up = x_rms @ self.model[f"model.layers.{layer}.mlp.up_proj.weight"].T
            ffn = (torch.nn.functional.silu(gate) * up) @ self.model[f"model.layers.{layer}.mlp.down_proj.weight"].T

            x = x + ffn
        return x
                

    def generate(self, messages):
        rendered = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        tokens = self.tokenizer(rendered,  return_tensors="pt", add_special_tokens=False).input_ids.to('mps')
        x = self.embed(tokens)

        generated_id = None
        with torch.no_grad():
            while generated_id == None or generated_id != self.tokenizer.eos_token_id:
                x = self.embed(tokens)

                x = self.forward_decode_layers(x)

                # Final norm + logits
                x_norm = torch.nn.functional.rms_norm(
                    x, normalized_shape=(self.config["hidden_size"],),
                    weight=self.model["model.norm.weight"],
                    eps=self.config["rms_norm_eps"],
                ).to(torch.bfloat16)

                logits = x_norm @ self.model["model.embed_tokens.weight"].T
                generated_id = int(logits[:, -1, :].argmax(dim=-1))
                if generated_id == self.tokenizer.eos_token_id:
                    break

                next_token = torch.tensor([[generated_id]], device=tokens.device, dtype=tokens.dtype)
                tokens = torch.cat([tokens, next_token], dim=1)
                
                print(self.tokenizer.decode(generated_id, skip_special_tokens=True), end='', flush=True)

Qwen2('./qwen2-0.5b').generate(
    [
        {"role": "user", "content": "Ouptut 1 a hundred times"},
    ]
)