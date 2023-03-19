import torch
from torch.optim import Adam
import scipy.io as sio
from scipy.sparse import coo_matrix
import numpy as np

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

# initialize M as zero matrix 
M = np.zeros((N,N))
for i in range(N):
    M[i,i] = 1
# for each pair in indices set value in M
for i in range(indices.shape[0]):
    M[indices[i,0], indices[i,1]] = values[0,i]
    M[indices[i,1], indices[i,0]] = values[0,i]

# compute real-valued eigen decomposition of symmetric matrix M
eigenvalues, eigenvectors = np.linalg.eigh(M)
# reconstruct M from eigen decomposition
M_reconstructed = np.dot(np.dot(eigenvectors, np.diag(eigenvalues)), eigenvectors.T)
# print reconstruction error
print("Reconstruction error: %g" % np.linalg.norm(M - M_reconstructed))



Dprime = 3
U = torch.nn.Parameter(torch.rand(N, Dprime))
S = torch.nn.Parameter(torch.rand(Dprime, 1)*2-1)

print(f"N: %d, Dprime: %d, values: %d" % (N, Dprime, values.squeeze().shape[0]))

# Simple dot product
def f(indices):
    I = indices[:,0]
    J = indices[:,1]
    
    EIJ = (U[I,:] @ torch.diag(S.squeeze())) * U[J,:]
    # row-wise sum of EIJ
    dij = torch.sum(EIJ, dim=1)
    return dij

# Loss function
def loss(indices, values):
    return torch.sum(torch.abs((f(indices) - torch.from_numpy(values)))**2)

print(f"MSE: %g" % (loss(indices, values)/N))

optimizer = Adam([U,S], lr=0.001)# define the loss function
# perform the optimization
num_epochs = 10000
print_iter = 100
print(f"%s | %10s %10s (%s,%s) (%s,%s)" % ("epoch", "loss", "L∞", "min(U)", "max(U)", "min(S)", "max(S)"))
for epoch in range(num_epochs):
    if epoch % print_iter == 0:
        print(f"%5d | %10g %10g (%g,%g) (%g,%g)" % (epoch,loss(indices, values)/N,torch.max(torch.abs(f(indices) - torch.from_numpy(values))),torch.min(U),torch.max(U),torch.min(S),torch.max(S)))
    optimizer.zero_grad()
    loss_value = loss(indices, values)
    loss_value.backward()
    optimizer.step()





