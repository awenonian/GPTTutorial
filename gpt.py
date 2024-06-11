# Setup was *way* easier if you use conda. Then it was basically just conda install cudatoolkit, and then the pytorch given command.
# It, *should* be working?

import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    # This is a self attention head
    def __init__(self, n_embd, head_size, block_size, dropout=.0):
        super().__init__()
        # I will explain what these end up doing in the "forward" function
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # For "self attention" we don't want future nodes to talk to previous nodes, so that they don't "give up the answer" of what comes next.
        # Therefore, we have to block out the keys/queries/values from future vectors with this "tril" (lower triangle) mask
        # (Not sure exactly what the "buffer" is, since the actual implementation is later, but I'm assuming it's to help torch save memory or something)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        # We're gonna have a bunch of these. They are a "regularization technique". 
        # Basically, at train time, the network will train with a bunch of weights set to 0, so the others can't rely on any one node. This helps overfitting
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        # The way to think about this is that the "key" is "here's what I contain"
        # The query is "Here's what I'm looking for"
        # If they are similar, we want to pay attention to that vector.
        k = self.key(x) # (B,T,C)
        q = self.query(x) # (B,T,C)
        # This essentially dot products the key and query values. 
        # It will be very positive if they point in the same direction, 0 if they're orthogonal, and very negative if they point in opposite directions
        # scale by 1/sqrt(channel)? I think this sorta normalizes the dot products, so they stay ~1
        # (weights), (@ is matrix mult), (we're transposing dimensions 2 and 3 (leaving them in the correct batches)
        # (for all these @s, I think real matrix mults wouldn't work on these dimensions (B,T,C), but torch is assuming the first one is batches, and T,C@C,T is valid)
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,C) @ (B,C,T) = (B,T,T)
        # As above comment, this is masking out future vectors. Normally, "tril" is 1s and 0s, but since we're gonna use softmax next, we need them to be -inf, so they'll translate into 0s
        # This is the line that makes it "self attention" 
        # If we were doing "cross attention" (which is used by stuff like translation, since the predicted tokens don't interact with the input tokens in the same way), we'd just remove this line.
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        # At this point, wei is the softmax of the dot products, or sort of a "what percent of my attention should be on this vector"
        # But, we now use the value matrix, which is essentially "if you think I'm important, here's what I'll tell you"
        # We can just multiply "how much we're paying attention" by the "what I want to tell you" to get some sort of "important info" vector
        v = self.value(x)
        out = wei @ v # (B, T, T) @ (B, T, C) = (B, T, C)
        return out

# This will be fairly simple
class MutliHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads, head_size, block_size, dropout=.0):
        super().__init__()
        # We want multiple heads, so make a list of multiple heads.
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size, dropout) for _ in range(num_heads)])
        # Not sure why this is necessary, when we'll do a linear in the ffwd part of block... Apparently it helps with the residual pathway (I think I'd put this in the Block code then, personally...)
        self.proj = nn.Linear(n_embd, n_embd)
        self. dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Result is then just, all the heads' results, concatenated.
        # Since these are (B,T,C), dim=-1 refers to C. This leaves Batches and Timing in-tact, giving (B,T,num_heads*C)
        # This is why in other places, we will divide n_embd by num_heads, to get head_size, so num_heads*C = n_embd
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

# Also fairly simple, feed forward. 
class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout=.0):
        super().__init__()
        # We put these in a "sequential" which I think is just the "handle this for me" if you just want to pass things through a series of layers
        self.net = nn.Sequential(
            # Basic layer
            nn.Linear(n_embd, 4*n_embd),
            # ReLU. I don't think this counts as a layer for purposes of .parameters() that we give the optimizer, but I think it's implemented as one, bc that's easiest?
            nn.ReLU(),
            # As above in MultiHead, this is the "projection" used for the residual pathway.
            # Above I said I'd implement it in the block, but here. But apparently, the GPT paper used a larger ffwd layer, so we're multiplying by 4. This makes sense to do in one place I guess
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Just run the sequence
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=.0):
        super().__init__()
        self.sa = MutliHeadAttention(n_embd, n_head, n_embd//n_head, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        # Layer norm, normalizes the layer. Apparently this helps, likely to keep things expanding too much.
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        # These "x +" bits are for the residual connections.
        # The original gpt paper had the norms happen after adding (self.ln1(x + self.sa(x))), but apparently doing it before is more common now
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class SelfAttentionGPT(nn.Module):
    # I've defaulted device to cpu, because that will always work, even though you *should* use a gpu, if you can.
    def __init__(self, vocab_size, n_embd, num_heads, num_blocks, block_size, dropout=.0, device="cpu"):
        super().__init__()
        # GPT Diagram (in ascii):
        # (at every step there is a "residual" connection, which is just an add and norm between the input to that step, and the output to that step.)
        #  This helps with optimization? Apparently explained in Andrej Karpathy's micrograd video, which I will watch later
        """                     Softmax ----> Output Probabilities
                                    /\ 
                                    |
                                Linear
                                    /\ 
                                ____|____________________________
                                |   |                           |
                                | Feed Forward                  |
                                |   /\                          |
                                |   |                           |
                --------------> | Multi Attention (w/o Mask)    |
        _______|_____________   |   /\                          | x N
        |      |            |   |   |                           |
        | Feed Forward      |   |   |                           |
    N x |    /\             |   |   |                           |
        |    |              |   |   |                           |
        | Multi Attention   |   | Multi Attention (w/ Mask)     |
        |___________________|   |_______________________________|
          /\                      /\ 
          |                       |
        Position                Position
        +                       +
        Input                   Output (shifted right)
        """
        # We will be skipping a lot of this, mostly the cross attention parts (meaning the "Input" section has no where to go, so that will be omitted too.)
        # If this was a cross attention thing, we'd include it, but it's self attention, so there's not much point.

        # Setup a lookup table that lets us look up the next token given the current token
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # Using our multihead thing. (// is "divide but drop the decimals")
        # Keeping the "num_heads" and divisor the same, means we'll get out the correct size
        # Might want to think about how you would implement this *without* this constraint, or, if you can't, then have it be built into the Module
        self.blocks = nn.Sequential(*[Block(n_embd, num_heads, block_size, dropout) for _ in range(num_blocks)])
        self.ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.device = device
        self.block_size = block_size

    def forward(self, idx, targets=None):
        # idx is the index of the current token (which will give us the next token)
        # targets is what the next token "should" be, from training
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) #(T,C)
        # torch tensors do some assumptions when you do matrix ops. 
        # + is only defined for matrices with the same dimensions, so when given (B,T,C)+(T,C), torch just assumes it's a batch thing, and copies (T,C) B times, to get a (B,T,C)
        x = tok_emb + pos_emb # (B,T,C)
       
        # Apply attention
        x = self.blocks(x)
        x = self.ln(x)

        logits = self.lm_head(x) # (B,T, vocab_size)

        if targets == None:
            loss = None
        else:
            # So, we'll be calculating the loss with cross entropy (good way to calc loss, has some nice properties, you can look up the math if you want)
            # https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html
            # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss
            # The problem is, the current shape of logts and targets is incorrect for the built in function.

            # logits is a (B,T,C) tensor, (stands for Batch, Time, Channels)
            # (Batch = 4, Time = 8, Channels = vocab_size)
            # This corresponds to our Batch size, our Block size, and our Vocab size.

            # cross_entropy wants (C) for unbatched single dimension input, and (minibatch,C,d1​,d2​,...,dK​) otherwise.
            # So, if we were doing one example at a time, and the input was a single token, then (C) would be appropriate (just the single output)
            # But ours is batched and multi-dimensional. minibatch is our batch size (4), and d1 is how many tokens we have in one example (8)

            # See below how we pass in xb, which is 4x8 and that's what idx is. So for each index in xb, we turn that into a whole (vocab_size) vector

            # Anyway, this means we need to reshape our logits, so we can use them correctly here.
            # We want to go from (B,T,C) -> (B,C,T)
            B, T, C = logits.shape
            # This shapes it so that instead of 4 batches of 8 (C) vectors, it's just one long list of 32 (C) vectors
            logitsv = logits.view(B*T, C)
            # Similarly, targets is (B,T), but now that logits is (B*T, C) they won't match up, so we need targets to be (B*T)
            targets = targets.view(B*T)

            # Compares what the lookup table said, to what the targets said
            loss = F.cross_entropy(logitsv, targets)

        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        # idx is (B,T), we want to take it to (B,T+max_new_tokens)
        for _ in range(max_new_tokens):
            # Previously, we cold put in as many tokens as we want, but now, we can't have more than block_size, or the positional embedding won't have an embedding for us.
            # So we'll just crop it to block_size. This discards everything but the last block_size tokens
            idx_cond = idx[:, -self.block_size:]

            # For modules, "self(...)" calls forward, I guess...
            logits, _ = self(idx_cond) # get logits
            logits = logits[:, -1, :] # just get the last character for each entry in batch (tensor of size (B,C))
            probs = F.softmax(logits, dim=-1) # idk what dim is, but this is softmax, it's just a standard thing with nice properties for turning tensors into probabilities.
            idx_next = torch.multinomial(probs, num_samples=1) # get one thing for each entry in batches (size (B,1))
            idx = torch.cat((idx, idx_next), dim=1) # Concat onto existing idx time (for each entry in batch) (size (B, T+1))
        return idx
    
