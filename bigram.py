import torch #importing pytorch (very important library for ml)
import torch.nn as nn
from torch.nn import functional as F #holds plain functions with no weights

# ---------------- hyperparameters ----------------
# [karpathy] all the settings in one place at the top, so you can change them without hunting through the code
batch_size = 32 #no. of chunks  # [karpathy] was 4 in the notebook
block_size = 8# size of chunks
max_iters = 12000 # number of training steps (your notebook value)
eval_interval = 300 # [karpathy] every 300 steps, print the train and val loss
learning_rate = 1e-3 #lr is your learning rate. (your notebook value)
# [karpathy] use the GPU if there is one: cuda = NVIDIA (RTX 5070), mps = Apple Silicon, otherwise cpu
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
eval_iters = 200 # [karpathy] how many batches to average over when estimating the loss
# -------------------------------------------------

torch.manual_seed(1337) # [karpathy] same random numbers every run, so results are reproducible

#importing text
with open('data/input.txt', 'r', encoding='utf-8') as f:
    text=f.read()

#sorting all 'characters' in the file
charss = sorted(list(set(text)))
vocab_size = len(charss)
 #we need to tokenize (convert these into integers)

string_to_integer = {} #defining a dictionary
for i in range(len(charss)):
    ch = charss[i] #assigning each character an integer
    string_to_integer[ch] = i #character is your key and the value is your position
#vice-versa
integer_to_string = {}
for i in range(len(string_to_integer)):
    integer_to_string[i] = charss[i]

#defining the encode and decode functions
def encode(s):
    result =[]
    for i in s: #goes through each character of the string
        result.append(string_to_integer[i])
    return result

def decode(i):
    result = []
    for c in i:
        result.append(integer_to_string[c])
    return ''.join(result) #joins all the characters into a string and returns it

#we are now going to tokenize the entire data.
data = torch.tensor(encode(text), dtype = torch.long) #storing the encoded text as a tensor in pytorch
#a tensor is an array, but unlike numpy arrays it can remember how its been changed and run on GPUs
#dtype is just defining the datatype.

#going on to seperate our data into a 'training' section and a 'validating' section
n = int(0.9 * len(data)) #first 90% of the data will be used to train our model
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    indices = torch.randint(len(data)-block_size, (batch_size,)) #first input defines max value, (subtract block size since we cant go outside the limit)
#batch size, we need 4 examples.
    #print(indices)#prints the randomly selected index from the data file.

    x = torch.stack([data[i:i+block_size]for i in indices])
    y = torch.stack([data[i+1:i+1+block_size]for i in indices])

    #torch.stack, 'stacks' all the tensors which we obtain after iterating through indices in a 2D tensor
    x, y = x.to(device), y.to(device) # [karpathy] move the batch to the same device as the model
    return x,y

# [karpathy] a less noisy loss: average over eval_iters batches, for both train and val
@torch.no_grad() # [karpathy] no backward() happens here, so tell PyTorch not to record steps for gradients (saves memory)
def estimate_loss():
    out = {}
    m.eval() # [karpathy] switch to evaluation mode (does nothing for this model yet, matters for later layers)
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train() # [karpathy] switch back to training mode
    return out

#logits are our list of raw unnormalized scores of the succeeding token (65 tokens - 65 scores - highest prob gets placed in bigram)
class BigramLangModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,vocab_size) #converts our tensor values by giving it rows of logits


    def forward(self,inputs, targets=None):
        logits = self.token_embedding_table(inputs) #creating logits (randomized at first) #(BATCH,TIME,CHANNEL) TENSOR
        if(targets is None):
            loss = None

        else:


            B,T,C = logits.shape #unpacking 4,8,65 into B T AND C
            logits = logits.view(B*T,C) #changing arrangement of array to 32*65 #created a 2D array
            #every single 'token' from our input will pluck out its row from the embedding table


            #now targets are in shape (b,t)2D ARRAY  i want to turn it into a 1D array of bt rows

            targets = targets.view(B*T)#4,8 to 32
            #we have fit logits and targets as per pytorch doc.

            loss = F.cross_entropy(logits,targets) #gives ur loss woohoo #cross entropy (simplified) = -log(probability) thats why initial loss is around -log(1/65)

        return logits,loss

    def generate(self,idx,max_token_value): #max_token_value = numbers
        for i in range(max_token_value):
            logits,loss = self(idx) #runs forward, returns just logits
            logits = logits[:,-1,:] #taking last digit of T ,shape is (B,C)
            probs = F.softmax(logits, dim =-1)#runs softmax function on the scores
            idx_next = torch.multinomial(probs, num_samples=1) #chooses the next token based on the probability, num samples means it will only take one
            idx = torch.cat((idx,idx_next), dim =1) #(B,T+1)

        return idx


m = BigramLangModel(vocab_size) #initializes a 65*65 embedding map
m = m.to(device) # [karpathy] move the model's weights (the embedding table) to the device

#creating a pytorch optimizer
#adam is a great and common optimizer used to train models (inteded for the same thing as SGD or just regular GD)
optimizer = torch.optim.AdamW(m.parameters(), lr = learning_rate) #lr is your learning rate.

#writing a typical training loop

for step in range(max_iters):

    # [karpathy] every eval_interval steps, print a reliable train and val loss
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb,yb = get_batch('train')
    logits, loss = m(xb,yb)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()
print(loss.item())

# [karpathy] the starting token has to be on the same device as the model
context = torch.zeros((1,1),dtype = torch.long, device = device)
print(decode(m.generate(idx = context,max_token_value=500)[0].tolist()))
