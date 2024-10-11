import torch
import torch.nn as nn
from torch.nn import functional as F

#  HYPERPARAMETER
EPOCHS = 10000
RANDOM_SEED = 1337
LEARNING_RATE = 3e-4
N_HEAD = 6

N_EMBD = 384
BLOCK_SIZE = 256
BATCH_SIZE = 64

dropout = 0.2
n_layer = 6
eval_iters = 200
eval_interval = 500
# ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(RANDOM_SEED)


def encode(s): return [stoi[c] for c in s]
def decode(e): return "".join([itos[i] for i in e])


with open("input.txt", "r", encoding="utf-8") as file:
    text = file.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
TRAIN_SPLIT = int(0.9*len(data))

train_data = data[:TRAIN_SPLIT]
val_data = data[TRAIN_SPLIT:]


def get_batch(split):
    """generates a batch of data as input x and output y"""

    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i: i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1: i+BLOCK_SIZE+1] for i in ix])
    x, y = x.to(device), y.to(device)

    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()

    return out


class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer("tril", torch.tril(
            torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, 16)
        q = self.query(x)  # (B, T, 16)
        v = self.value(x)

        # (B, T, 16) @ (B, 16, T) --> (B, T, T)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=1)
        wei = self.dropout(wei)
        out = wei @ v

        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(N_EMBD, N_EMBD)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.projection(out))

        return out


class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLM(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(
            *[Block(N_EMBD, n_head=N_HEAD) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, context, targets=None):

        # context shape = batch x time x channel (B, T, C)
        B, T = context.shape
        token_emb = self.token_embedding_table(context)  # (B,T,C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device))  # (T, C)
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocal_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, context, max_new_tokens):
        #  context is of shape (B, T)
        for _ in range(max_new_tokens):
            cropped_context = context[:, -BLOCK_SIZE:]
            logits, loss = self(cropped_context)
            logits = logits[:, -1, :]  # (B,C)

            probs = F.softmax(logits, dim=1)
            next_context = torch.multinomial(probs, num_samples=1)  # (B, 1)
            context = torch.cat((context, next_context), dim=1)  # (B, T+1)
        return context


model = BigramLM()
m = model.to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):

    if epoch % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {epoch}: train loss:{losses['train']:.4f}, val loss:{losses['val']:.4f}")

    xb, yb = get_batch("train")

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
