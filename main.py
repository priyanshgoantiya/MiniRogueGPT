# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import dropout

torch.manual_seed(1337)
# Define hyperparameters
batch_size = 64
block_size = 256  # Context length for training
max_iters=5000
eval_interval=500
eval_iter=200
lr=3e-4
num_embd=384
dropout=0.2
n_head = 6
n_layer = 6
#--------------------------------------------------------------------------------------------------------------
torch.manual_seed(1337)
# Load dataset
with open('input.txt', 'r', encoding='utf-8') as file:
  text = file.read()

# print(f'Length of dataset characters: {len(text)}')

# Create vocabulary
chars = sorted(set(text))
vocab_size = len(chars)
# print(f'Vocabulary size: {vocab_size}')

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # Encode string to integer list
decode = lambda l: ''.join([itos[i] for i in l])  # Decode integer list to string

# Encode dataset into tensor
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # Split 90% train, 10% test
training_data, test_data = data[:n], data[n:]




def get_batch(split):
  data = training_data if split == 'train' else test_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  X = torch.stack([data[i:i + block_size] for i in ix])
  y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
  return X, y

@torch.no_grad()
def loss_estimate():
  out={ }
  m.eval()
  for split in ['train','test']:
    losses=torch.zeros(eval_iter)
    for k in range(eval_iter):
      X, y = get_batch('train')
      logits, loss = m(X, y)
      losses[k]=loss.item()
    out[split]=losses.mean()
  m.train()
  return out
# head_size=16 # hyper parameter
class Head(nn.Module):
  """ One Head of the self attention"""
  def __init__(self,head_size):
    super().__init__()
    self.key = nn.Linear(num_embd, head_size, bias=False)
    self.query = nn.Linear(num_embd, head_size, bias=False)
    self.value = nn.Linear(num_embd, head_size, bias=False)
    self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
    self.dropout=nn.Dropout(dropout)
  def forward(self,X):
    B,T,C=X.shape
    k = self.key(X)  # (B,T,16)
    q = self.query(X)  # (B,T,16)
    wei = q @ k.transpose(-2, -1)*C**-0.5  # (B,T,16) @ (B,16,T) = (B,T,T)
    wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) #(B,T,T)
    wei = torch.softmax(wei, dim=-1)
    wei=self.dropout(wei)
    v = self.value(X)
    out = wei @ v
    return out

class MultiHeadAttention(nn.Module):
  """Multi Head of the self attention in parallel"""
  def __init__(self,num_head,head_size):
    super().__init__()
    self.heads=nn.ModuleList([Head(head_size) for _ in range(num_head)])
    self.projection=nn.Linear(num_embd,num_embd)
    self.dropout = nn.Dropout(dropout)
  def forward(self,X):
    out=torch.cat([h(X) for h in self.heads],dim=-1)
    out=self.dropout(self.projection(out))
    return out
class FeedForward(nn.Module):
  """ A simple linear layer followed by a non-linearity"""
  def __init__(self,num_embd):
    super().__init__()
    self.net=nn.Sequential(nn.Linear(num_embd,4*num_embd),nn.ReLU(),nn.Linear(4*num_embd,num_embd),
                           nn.Dropout(dropout),)
  def forward(self,X):
    return self.net(X)

class Block(nn.Module):
  """ Transformer block : communication followed by computation"""
  def __init__(self,num_embd,n_head):
    super().__init__()
    head_size=num_embd//n_head
    self.sa=MultiHeadAttention(n_head,head_size)
    self.ffwd=FeedForward(num_embd)
    self.layer_norm1=nn.LayerNorm(num_embd)
    self.layer_norm2 = nn.LayerNorm(num_embd)
  def forward(self,X):
    X=X+self.sa(self.layer_norm1(X))
    X= X + self.ffwd(self.layer_norm2(X))
    return X


# Define the model
class BigramLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size,num_embd)
    self.position_embedding_table = nn.Embedding(block_size, num_embd)
    self.Blocks=nn.Sequential(*[Block(num_embd,n_head=n_head) for _ in range(n_layer)])
    self.layer_norm_final=nn.LayerNorm(num_embd)
    self.lm_head=nn.Linear(num_embd,vocab_size)
  def forward(self, idx, targets=None):
    B,T=idx.shape
    tok_embd= self.token_embedding_table(idx)  # (B,T,C)
    pos_embd=self.position_embedding_table(torch.arange(T))## (T,C)
    contextual_representation=tok_embd+pos_embd
    contextual_representation=self.Blocks(contextual_representation)
    logits=self.lm_head(contextual_representation) #(B,T, vocab_size)
    loss = None
    if targets is not None:
      batch_size, block_size, vocab_size = logits.shape
      logits = logits.view(batch_size * block_size, vocab_size)
      targets = targets.view(batch_size * block_size)
      loss = F.cross_entropy(logits, targets)
    return logits, loss

  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      idx_cond=idx[:,-block_size:]
      logits, _ = self(idx_cond)
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)
    return idx


# Initialize model and optimizer
m = BigramLanguageModel()
optimizer = torch.optim.AdamW(m.parameters(), lr=lr)

for iter in range(max_iters):
  if iter % eval_interval==0 or iter == max_iters - 1:
    losses=loss_estimate()
    print(f'step {iter}: train loss {losses['train']:.4f},test loss {losses['test']:.4f}')

  X,y=get_batch('train')
  X, y = get_batch('train')
  logits, loss = m(X, y)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

# Generate text
context=torch.zeros((1, 1), dtype=torch.long)
generated_text=decode(m.generate(context, max_new_tokens=10000)[0].tolist())

with open('more.txt','w',encoding='utf-8') as file:
  file.write(generated_text)