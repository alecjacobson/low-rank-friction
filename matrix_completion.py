import torch
from torch.optim import Adam
import scipy.io as sio

torch.manual_seed(0)

# read indices, values, materials from a .mat file
def read_mat_file(file_name):
    data = sio.loadmat(file_name)
    indices = data['indices']
    values = data['values']
    materials = data['materials']
    return indices, values, materials


(indices, values, materials) = read_mat_file('data.mat')


# convert indices, values to list of triplets
def convert_to_triplets(indices, values):
    triplets = []
    for i in range(indices.shape[0]):
        triplets.append((indices[i,0], indices[i,1], values[0,i]))
    return triplets

triplets = convert_to_triplets(indices, values)

# number of materials
N = indices.max() + 1

U = torch.nn.Parameter(torch.rand(N, 1))

# Simple dot product
def f(U, indices):
    I = indices[:,0]
    J = indices[:,1]
    EIJ = U[I,:] * U[J,:]
    # row-wise sum of EIJ
    return torch.sum(EIJ, dim=1)

# Loss function
def loss(U, indices, values):
    return torch.sum(torch.abs((f(U, indices) - torch.from_numpy(values)))**2)

print(f"MSE: %g" % (loss(U, indices, values)/N))

optimizer = Adam([U], lr=0.04)# define the loss function
# perform the optimization
num_epochs = 1000
print_iter = 100
print(f"%s | %10s %10s (%s,%s)" % ("epoch", "loss", "L∞", "min(U)", "max(U)"))
for epoch in range(num_epochs):
    if epoch % print_iter == 0:
        print(f"%5d | %10g %10g (%g,%g)" % (epoch,loss(U, indices, values)/N,torch.max(torch.abs(f(U,indices) - torch.from_numpy(values))),torch.min(U),torch.max(U)))
    optimizer.zero_grad()
    loss_value = loss(U, indices, values)
    loss_value.backward()
    optimizer.step()





