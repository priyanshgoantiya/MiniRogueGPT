import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------
# Setup
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load vocabulary (same preprocessing as training)
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

# -----------------------
# Define Model
# -----------------------
block_size = 128
num_embd = 256
n_head = 4
n_layer = 4
dropout = 0.2

class TransformerBlock(nn.Module):
    def __init__(self, num_embd, n_head):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=num_embd, num_heads=n_head, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(num_embd, 4 * num_embd),
            nn.ReLU(),
            nn.Linear(4 * num_embd, num_embd),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(num_embd)
        self.norm2 = nn.LayerNorm(num_embd)

    def forward(self, X):
        X = X + self.attn(self.norm1(X), self.norm1(X), self.norm1(X))[0]
        X = X + self.ff(self.norm2(X))
        return X

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, num_embd)
        self.pos_embedding = nn.Embedding(block_size, num_embd)
        self.blocks = nn.Sequential(*[TransformerBlock(num_embd, n_head) for _ in range(n_layer)])
        self.norm = nn.LayerNorm(num_embd)
        self.head = nn.Linear(num_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_embd = self.token_embedding(idx)
        pos_embd = self.pos_embedding(torch.arange(T, device=device))
        X = tok_embd + pos_embd
        X = self.blocks(X)
        logits = self.head(self.norm(X))
        return logits, None

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# -----------------------
# Load trained weights
# -----------------------
model = BigramLanguageModel().to(device)
model.load_state_dict(torch.load("language_model_step4999.pth", map_location=device))
model.eval()

# -----------------------
# Streamlit UI
# -----------------------
st.title("📝 Text Generation with Transformer")
st.write("This app generates text using your trained language model.")

prompt = st.text_input("Enter a prompt:", "Once upon a time")
max_tokens = st.slider("Number of tokens to generate:", 50, 1000, 200)

if st.button("Generate"):
    context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=max_tokens)
    output_text = decode(generated[0].tolist())
    st.text_area("Generated Text:", output_text, height=300)
