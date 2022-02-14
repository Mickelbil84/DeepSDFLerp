import os

import tqdm
import torch
import visdom
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from constants import *

import data
import model

# Init Visdom
vis = None
if __name__ == "__main__":
    vis = visdom.Visdom(env='deepsdf')

# Make sure checkpoints folder exists
if not os.path.isdir('checkpoints'):
    os.mkdir('checkpoints')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(train_loader, embedding, model, criterion, optimizer, optimizer_embedding, epoch, train_losses):
    total_loss = 0
    cnt = 0
    train_enum = tqdm.tqdm(enumerate(train_loader),
                           desc='Train epoch %d' % epoch)

    for i, data in train_enum:
        # Get xyz,latent as input and sdf and mesh index
        xyz = data['xyz'].to(device)
        latent_idx = data['idx']
        sdf = data['sdf'].to(device)

        # Fetch latent vector from embedding
        latent = embedding(latent_idx.to(device))

        # Perform a learning step
        model.zero_grad()
        embedding.zero_grad()
        output = model(xyz, latent)
        loss = criterion(torch.clamp(output, -DELTA, DELTA),
                         torch.clamp(sdf, -DELTA, DELTA))
        loss.backward()
        optimizer.step()
        optimizer_embedding.step()

        # Update losses and visualize
        total_loss += loss.item()
        cnt += 1
        if i % 1000 == 0:
            train_enum.set_description(
                'Train (loss %.8f) epoch %d' % (loss.item(), epoch))
            train_losses.append(loss.item())
            vis.line(Y=np.asarray(train_losses), X=torch.arange(1, 1+len(train_losses)),
                     opts={'title': 'Loss'}, name='loss', win='loss')

    # Print summary of epoch
    print('=====> Train set loss: {:.8f}\t Epoch: {}'.format(
        total_loss / cnt, epoch))


def main():
    ###################
    # Load train data
    ###################
    train_dataset = data.DeepSDFDataset()
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS, pin_memory=True, shuffle=True)

    ###############################
    # Prepare model and optimizers
    ###############################
    deepsdf = model.DeepSDF().to(device)
    embedding = model.LatentEmbedding(NUM_MESHES).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(deepsdf.parameters(), lr=LR_SDF)
    optimizer_embedding = optim.Adam(embedding.parameters(), lr=LR_EMBEDDING)

    ###############################
    # Start training
    ###############################
    train_losses = []
    for epoch in range(NUM_EPOCHS):
        train(train_loader, embedding, deepsdf, criterion,
              optimizer, optimizer_embedding, epoch, train_losses)
        torch.save(deepsdf.state_dict(),
                   'checkpoints/checkpoint_e%d.pth' % epoch)
        torch.save(embedding.state_dict(),
                   'checkpoints/checkpoint_e%d_emb.pth' % epoch)


if __name__ == "__main__":
    main()
