import torch  # Import PyTorch library
import torch.nn as nn  # Import neural network module
import torch.nn.functional as F  # Import functional module for additional operations

torch.manual_seed(1337)  # Set seed for reproducibility
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available, otherwise CPu

# Optimized Hyperparameters
batch_size = 32  # Number of samples per training batch
block_size = 128  # Context window size
max_iters = 2000  # Total number of training iterations
eval_interval = 500  # Interval for evaluating loss
eval_iter = 100  # Number of iterations to compute average loss
lr = 3e-4  # Learning rate for optimizer
num_embd = 256  # Embedding size
dropout = 0.2  # Dropout rate to prevent overfitting
n_head = 4  # Number of attention heads
n_layer = 4  # Number of transformer layers

# Load dataset
with open('input.txt', 'r', encoding='utf-8') as file:
    text = file.read()  # Read entire text file

# Create vocabulary
chars = sorted(set(text))  # Get unique characters in sorted order
vocab_size = len(chars)  # Number of unique characters

stoi = {ch: i for i, ch in enumerate(chars)}  # Character to index mapping
itos = {i: ch for i, ch in enumerate(chars)}  # Index to character mapping
encode = lambda s: [stoi[c] for c in s]  # Convert string to list of indices
decode = lambda l: ''.join([itos[i] for i in l])  # Convert list of indices back to string

# Encode dataset into tensor
data = torch.tensor(encode(text), dtype=torch.long, device=device)  # Convert text to tensor
n = int(0.9 * len(data))  # 90% of data for training
training_data, test_data = data[:n], data[n:]  # Split data into training and test sets

# Function to get batches
def get_batch(split):
    data = training_data if split == 'train' else test_data  # Select dataset
    ix = torch.randint(len(data) - block_size, (batch_size,), device=device)  # Random start indices
    X = torch.stack([data[i:i + block_size] for i in ix])  # Create input batches
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])  # Create target batches
    return X, y  # Return input-target pairs

@torch.no_grad()
def loss_estimate():
    out = {}  # Dictionary to store loss values
    m.eval()  # Set model to evaluation mode
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iter, device=device)  # Initialize loss tensor
        for k in range(eval_iter):
            X, y = get_batch(split)  # Get batch
            _, loss = m(X, y)  # Compute loss
            losses[k] = loss.item()  # Store loss value
        out[split] = losses.mean().item()  # Compute mean loss
    m.train()  # Set model back to training mode
    return out  # Return loss values

# Define Transformer Blocks
class TransformerBlock(nn.Module):
    def __init__(self, num_embd, n_head):
        super().__init__()
        head_size = num_embd // n_head  # Compute size of each attention head
        self.attn = nn.MultiheadAttention(embed_dim=num_embd, num_heads=n_head, dropout=dropout, batch_first=True)  # Multi-head attention layer
        self.ff = nn.Sequential(
            nn.Linear(num_embd, 4 * num_embd), nn.GELU(),  # Feed-forward expansion and activation
            nn.Linear(4 * num_embd, num_embd), nn.Dropout(dropout)  # Reduce dimensions and apply dropout
        )
        self.norm1 = nn.LayerNorm(num_embd)  # Layer normalization for attention
        self.norm2 = nn.LayerNorm(num_embd)  # Layer normalization for feed-forward
        self.dropout=nn.Dropout(dropout)

    def forward(self, X):
        T = X.size(1)
        mask = torch.triu(torch.ones(T, T, device=X.device), diagonal=1).bool()
        attn_out, _ = self.attn(self.norm1(X), self.norm1(X), self.norm1(X), attn_mask=mask)
        X = X + self.dropout(attn_out)
        X = X + self.ff(self.norm2(X))
        return X  # Return transformed output

# Define the model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, num_embd)  # Token embeddings
        self.pos_embedding = nn.Embedding(block_size, num_embd)  # Positional embeddings
        self.dropout=nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[TransformerBlock(num_embd, n_head) for _ in range(n_layer)])  # Stack transformer blocks
        self.norm = nn.LayerNorm(num_embd)  # Final layer normalization
        self.head = nn.Linear(num_embd, vocab_size)  # Final linear layer to predict logits

    def forward(self, idx, targets=None):

        B, T = idx.shape  # Get batch size and sequence length
        tok_embd = self.token_embedding(idx)  # Compute token embeddings
        pos_embd = self.pos_embedding(torch.arange(T, device=device))  # Compute positional embeddings
        X = tok_embd + pos_embd  # Combine embeddings
        X=self.dropout(X)
        X = self.blocks(X)  # Pass through transformer blocks
        logits = self.head(self.norm(X))  # Compute final logits

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))  # Compute loss if targets provided

        return logits, loss  # Return logits and loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]  # Get last block_size tokens as context
            logits, _ = self(idx_cond)  # Get logits
            logits = logits[:, -1, :]  # Select last time step
            probs = F.softmax(logits, dim=-1)  # Convert logits to probabilities
            idx_next = torch.multinomial(probs, num_samples=1)  # Sample next token
            idx = torch.cat((idx, idx_next), dim=1)  # Append to generated sequence
        return idx  # Return generated sequence

# Initialize model and optimizer
m = BigramLanguageModel().to(device)  # Move model to device
optimizer = torch.optim.AdamW(m.parameters(), lr=lr)  # Define optimizer

# Training loop
for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = loss_estimate()  # Compute loss
        print(f'step {iter}: train loss {losses["train"]:.4f}, test loss {losses["test"]:.4f}')  # Print loss
        torch.save(m.state_dict(), f"language_model_step{iter}.pth")
    X, y = get_batch('train')  # Get training batch
    logits, loss = m(X, y)  # Forward pass
    optimizer.zero_grad(set_to_none=True)  # Zero gradients
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights

# # Save the model
# torch.save(m.state_dict(), "language_model.pth")  # Save model parameters

# Generate text
context = torch.zeros((1, 1), dtype=torch.long, device=device)  # Start with empty context
generated_text = decode(m.generate(context, max_new_tokens=1000)[0].tolist())  # Generate text

# Save generated text
with open('generated_text.txt', 'w', encoding='utf-8') as file:
    file.write(generated_text)  # Save generated text to file