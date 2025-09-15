import random
from typing import List, Optional, Tuple
import torch
from transformers import CLIPImageProcessor, AutoTokenizer
import torch
import json
from safetensors.torch import load_file 

# RoPE
def rotate_half(t: torch.Tensor) -> torch.Tensor:
    d2 = t.shape[-1] // 2
    return torch.cat((-t[..., d2:], t[..., :d2]), dim=-1)

def apply_rope(qk: torch.Tensor, base: float, token_idx: int) -> torch.Tensor:
    _, _, S, D = qk.shape
    assert D % 2 == 0
    inv = torch.arange(0, D, 2, device=qk.device, dtype=torch.float32) / D
    inv_freq = base ** (-inv) # (D/2,)
    t = torch.arange(token_idx, token_idx+S, device=qk.device, dtype=torch.float32) # (S,)
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

        # Set KV Cache
        SEQ_LEN = 150
        self.kv_cache = [{
            "key": torch.empty((1, self.config["num_attention_heads"], SEQ_LEN, self.config["hidden_size"] // self.config["num_attention_heads"]), dtype=torch.bfloat16, device=self.device),
            "value": torch.empty((1, self.config["num_attention_heads"], SEQ_LEN, self.config["hidden_size"] // self.config["num_attention_heads"]), dtype=torch.bfloat16, device=self.device),
        } for _ in range(self.config["num_hidden_layers"])]

    def forward(self, tokens, token_count):
        x = self.embed(tokens)
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
            k = k.view(1, S, n_kv, head_dim).transpose(1, 2) # [1,KV,S,D]
            v = v.view(1, S, n_kv, head_dim).transpose(1, 2)

            if n_heads != n_kv:
                reps = n_heads // n_kv
                k = k.repeat_interleave(reps, dim=1)
                v = v.repeat_interleave(reps, dim=1)

            # RoPE
            theta = self.config["rope_theta"]
            q = apply_rope(q, theta, token_idx=token_count)
            k = apply_rope(k, theta, token_idx=token_count)

            # KV cache
            self.kv_cache[layer]["key"][:, :, token_count:token_count+S] = k # B, H, s:s+x, D
            self.kv_cache[layer]["value"][:, :, token_count:token_count+S] = v # B, H, s:s+x, D

            k = self.kv_cache[layer]["key"][:, :, :token_count+S]
            v = self.kv_cache[layer]["value"][:, :, :token_count+S]

            # attn
            scores = (q @ k.transpose(-2, -1)) / (head_dim ** 0.5) # [1,H,S,S]
            causal = torch.triu(torch.full((S, S + token_count), float("-inf"), device='mps', dtype=scores.dtype), 1 + token_count) # mask
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

        # final norm + logits
        x_norm = torch.nn.functional.rms_norm(
            x, normalized_shape=(self.config["hidden_size"],),
            weight=self.model["model.norm.weight"],
            eps=self.config["rms_norm_eps"],
        ).to(torch.bfloat16)

        logits = x_norm @ self.model["model.embed_tokens.weight"].T
        return torch.softmax(logits, dim=-1)

    def tokenize(self, messages):
        tokens = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to('mps')
        return tokens

    def decode(self, tokens):
        return [self.tokenizer.decode(id, skip_special_tokens=True) for id in tokens]
    
    def sample_from_logits(self, logits):
        """ greedy sampling, return a single int token id """
        return int(logits[:, -1, :].argmax(dim=-1).item())

    def generate(self, tokens, n=128, return_logits=False, print_stream=False, cursor=0) -> Tuple[torch.Tensor, torch.Tensor]:
        generated_id = None
        processed_tokens = 0
        generated_tokens = 0

        # store tokens as [1, n] sequence
        output_tokens = torch.empty((1, 0), device=tokens.device, dtype=torch.long)
        # store logits as [1, n, vocab]
        output_logits = torch.empty((1, 0, self.config["vocab_size"]), device=tokens.device, dtype=torch.bfloat16)

        with torch.no_grad():
            while (generated_id is None or generated_id != self.tokenizer.eos_token_id) and generated_tokens < n:
                # forward pass
                logits = self.forward(tokens, token_count=cursor+processed_tokens)

                # sample next id
                generated_id = self.sample_from_logits(logits)
                if generated_id == self.tokenizer.eos_token_id:
                    break

                next_token = torch.tensor([[generated_id]], device=tokens.device, dtype=torch.long)

                # update counters
                processed_tokens += tokens.shape[1]
                generated_tokens += 1
                tokens = next_token.clone()

                # print incremental decode
                if print_stream:
                    print(self.tokenizer.decode([generated_id], skip_special_tokens=True), end='', flush=True)

                # append token + last-step logits
                output_tokens = torch.cat([output_tokens, next_token], dim=-1)
                if return_logits:
                    output_logits = torch.cat([output_logits, logits], dim=1)

        return output_tokens, output_logits


# model
draft = Qwen2('./qwen2-0.5b')
target = Qwen2('./qwen2-1.5b')

# tokenize
draft_n = 4
tokens = draft.tokenize([
    {"role": "user", "content": "Describe a shakespeare skit in detail."}
])
initial_len = tokens.shape[1]
cursor = 0

while (tokens.shape[1] - initial_len) < 100:
    # draft 4 tokens, get logits
    output_draft_tokens, ouput_draft_logits = draft.generate(tokens, n=draft_n, return_logits=True, cursor=cursor)

    # run large model
    verify_tokens = torch.cat([tokens, output_draft_tokens], dim=-1)
    output_target_tokens, ouput_target_logits = target.generate(verify_tokens, n=1, return_logits=True, cursor=cursor)

    cursor += tokens.shape[1]

    # for each new token logits
    n = 0
    accept = True
    accepted_tokens = torch.empty((1, 0), dtype=tokens.dtype, device=tokens.device)
    while n < draft_n and accept:
        draft_token = output_draft_tokens[:, [-(draft_n-n)]]
        draft_token_idx = draft_token[0].item()
        draft_logits = ouput_draft_logits[0, -(draft_n-n)]
        qx = draft_logits[draft_token_idx].item() # Q(x) from paper
        new_token_from_target = None

        target_logits = ouput_target_logits[0, -(draft_n-n)-1] # remember target has one more logits than draft as we infer on top of draft tokens
        px = target_logits[draft_token_idx].item() # P(x) from papers

        if (qx < px): # if target model is more confident than draft, accept token
            accept = True # redundant, left here for explicability
            accepted_tokens = torch.cat([accepted_tokens, draft_token], dim=-1)
        else:
            # draft is more confident than target
            if (random.random() < px / qx): # accept with probability px/qx (since px < qx, px/qx is a fraction) 
                accept = True # redundant, left here for explicability
                accepted_tokens = torch.cat([accepted_tokens, draft_token], dim=-1)
            else:
                accept = False # reject any further tokens from draft model
                
                # instead sample as follows and accept that token
                residual = torch.clamp(target_logits - draft_logits, min=0.0)
                residual = residual / residual.sum()  # normalize
                new_token = torch.multinomial(residual, num_samples=1)
                new_token_from_target = new_token.unsqueeze(0)

        n += 1

    if accept == True: # all tokens were accepted
        new_token_from_target = output_target_tokens

    print(draft.decode(accepted_tokens)[0], end='', flush=True)
    print(draft.decode(new_token_from_target)[0], end='', flush=True)
    cursor += accepted_tokens.shape[1]
    tokens = new_token_from_target