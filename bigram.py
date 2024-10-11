import torch
import torch.nn as nn
from torch.nn import functional as F

#  HYPERPARAMETER
EPOCHS = 5000
RANDOM_SEED = 1337
LEARNING_RATE = 1e-3

block_size = 8
batch_size = 32
eval_iters = 200
eval_interval = 300
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
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i: i+block_size] for i in ix])
    y = torch.stack([data[i+1: i+block_size+1] for i in ix])
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


class BigramLM(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, context, targets=None):
        logits = self.token_embedding_table(
            context)  # batch x time x channel (B,T,C)

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
            logits, loss = self(context)
            logits = logits[:, -1, :]  # (B,C)

            probs = F.softmax(logits, dim=1)
            next_context = torch.multinomial(probs, num_samples=1)  # (B, 1)
            context = torch.cat((context, next_context), dim=1)  # (B, T+1)
        return context


model = BigramLM(vocab_size)
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
