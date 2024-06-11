import os

import torch

from gpt import SelfAttentionGPT

# Defining a manual seed should make things reproducible. I didn't test this though, only ran each version once.
torch.manual_seed(1234)

out_dir = 'out'

# So this is used to run on GPU. You'll see lots of .to(device) type things throughout (and a device=device at generate)
# Those are necessary, keep an eye out and remember that
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device != 'cuda':
    print("Running on CPU :(")

batch_size = 64
block_size = 256

max_iters = 5000
eval_interval = 500
# How many iterations get run in an evaluation
eval_iters = 200
learning_rate = 1e-3
n_embd = 384
num_blocks = 6
num_heads = 6
dropout = .2

# In nanoGPT, he has logic here to determine if this is the "master process" which is used in distributed compute
# That's really cool, and you can check it out, but I'm only running this locally.
os.makedirs(out_dir, exist_ok=True)

# Grab shakespeare data set
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Do an encoding. This is simple (assign an index to each char, in ascii order), but maybe not smart
# OpenAI apparently uses "tiktoken" library to do it's encoding
# Google uses SentencePiece
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

stoi = { ch:i for i,ch in enumerate(chars)}
itos = { i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Turn it into a tensor to be consumed by model
data = torch.tensor(encode(text), dtype=torch.long)

# Split to train vs test
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# Split training into blocks, so it can be done in chunks

# split is either "train" or "test"
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

xb, yb = get_batch('train')

model = SelfAttentionGPT(vocab_size, n_embd, num_heads, num_blocks, block_size, dropout, device)
m = model.to(device)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)


# So here we're making this Adam optimizer (standard thing with nice properties (it modulates learning rate per parameter))
# lr is our learning rate, apparently e-4 is more often, but this is simple so maybe can be done with higher rates
# we pass in m.parameters() which I guess the model assumes based on what properties it has assigned (in our case, just the Embedding)? 
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)


# This is something he has in his video, many notes about it. It's not in the notebook tho
# This says "don't calculate gradients on this" because we won't call that .backward function.
# I still don't quite understand how the .backward function knows what to do, though, so this didn't answer my question
@torch.no_grad()
def estimate_loss():
    out = {}
    # We put the model in eval mode, which won't do anything *right now*, but *could* do something
    # And it's important to do our loss estimate in eval instead of training for this reason.
    model.eval()
    # Why are we doing this on both, not just val?
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            # Just want loss
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

for steps in range(max_iters):
    # Get a batch
    xb, yb = get_batch('train')

    # Predict for it, plus losses
    logits, loss = m(xb, yb)
    # This clears the previous gradients (which the parameters (see above) store themselves)
    optimizer.zero_grad(set_to_none=True)
    # A large confusion I have is over this line. I'll just say for now that it works, so if you're not curious about the inner workings, then it doesn't matter.
    # But the question I have is how does this loss know what to update? We calculated this in the "forward" function of our model as the cross entropy between
    # our expected outputs and our actual outputs, which are both just lists of numbers (well, tensors). My best guess is that tensors, if you don't wrap the function in
    # "@torch.no_grad()" keep a list of their history, and the sort of "blame" for their decisions, so then this .backward() can know about them. But I'm not sure.
    loss.backward()
    # This updates all parameters based on their gradients and whatever method we're using (AdamW)
    optimizer.step()
    # every interval, or on the last run, do an eval
    if steps % eval_interval == 0 or steps == max_iters-1:
        losses = estimate_loss()
        print(f"step {steps}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")


# Here's how you might generate with this
# This makes a 1x1 with a 0 in it (probably a newline character). 1x1 is the batch and time dimensions from above.
indexes = torch.zeros((1,1), dtype=torch.long, device=device)
# The [0] is needed because the data comes back in batches (in this case a batch of 1 but that's still something to be indexed into)
print(decode(m.generate(indexes, max_new_tokens=100)[0].tolist()))

with open("output.txt", "w") as file:
    file.write(decode(m.generate(indexes, max_new_tokens=10000)[0].tolist()))