"""
Targeted diagnostic: test SDPA backends for NaN root cause.
Run: CUDA_VISIBLE_DEVICES=1 python -m scripts_silver_cleaning.diagnose_neodictabert
"""
import torch
import transformers

CKPT = "/path/to/Hebrew_NLI/finetune_heb_nli/outputs/dicta-il__neodictabert/checkpoint-5600"

def load_model():
    tokenizer = transformers.AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
    config    = transformers.AutoConfig.from_pretrained(CKPT, trust_remote_code=True)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        CKPT, config=config, trust_remote_code=True, torch_dtype=torch.float32,
    )
    model.to("cuda:0").eval()
    return tokenizer, model

def run_pair(tokenizer, model, label=""):
    premise = "ברק אובמה נולד בהוואי."
    hypo    = "מקום הלידה של ברק אובמה הוא הוואי."
    enc = tokenizer(premise, hypo, return_tensors="pt", truncation=True,
                    add_special_tokens=True, return_token_type_ids=False)
    for k in enc:
        enc[k] = enc[k].to("cuda:0")
    with torch.no_grad():
        logits = model(**enc, return_dict=True).logits
    nan = torch.isnan(logits).any().item()
    print(f"  [{label}] logits={logits.tolist()}  NaN={nan}")
    return not nan

tokenizer, model = load_model()

# --- Test 1: default (Flash + MemEfficient SDPA) ---
print("\n=== Test 1: Default SDPA (Flash + MemEfficient enabled) ===")
run_pair(tokenizer, model, "default")

# --- Test 2: disable Flash, keep MemEfficient ---
print("\n=== Test 2: Flash disabled, MemEfficient enabled ===")
torch.backends.cuda.enable_flash_sdp(False)
run_pair(tokenizer, model, "no-flash")

# --- Test 3: disable both Flash and MemEfficient → math backend only ---
print("\n=== Test 3: Math backend only (Flash+MemEfficient disabled) ===")
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
run_pair(tokenizer, model, "math-only")

# --- Test 4: restore all, use eager attention (output_attentions=True) ---
print("\n=== Test 4: Eager attention (output_attentions=True) ===")
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
enc4 = tokenizer("ברק אובמה נולד בהוואי.", "מקום הלידה של ברק אובמה הוא הוואי.",
                 return_tensors="pt", truncation=True, add_special_tokens=True, return_token_type_ids=False)
for k in enc4:
    enc4[k] = enc4[k].to("cuda:0")
with torch.no_grad():
    out4 = model(**enc4, return_dict=True, output_attentions=True)
print(f"  [eager] logits={out4.logits.tolist()}  NaN={torch.isnan(out4.logits).any().item()}")

# --- Test 5: check available SDPA backends ---
print("\n=== SDPA backend availability ===")
print(f"  Flash SDP enabled    : {torch.backends.cuda.flash_sdp_enabled()}")
print(f"  MemEfficient enabled : {torch.backends.cuda.mem_efficient_sdp_enabled()}")
print(f"  Math SDP enabled     : {torch.backends.cuda.math_sdp_enabled()}")

print("\n=== Done ===")
