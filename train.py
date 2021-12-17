import tqdm
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import data
import model

if __name__ == "__main__":
    # Load chain data
    train_dataset = data.DeepSDFDataset()
    train_loader = DataLoader(train_dataset, batch_size=128, num_workers=4, pin_memory=True, shuffle=True)

    deepsdf = model.DeepSDF().cuda()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(deepsdf.parameters())

    for epoch in range(1):
        for i, data in tqdm.tqdm(enumerate(train_loader)):
            xyz = data['xyz'].float().cuda()
            latent = data['latent'].float().cuda()
            sdf = torch.from_numpy(np.array([sdf_labels[i]])).float().cuda()

            optimizer.zero_grad()

            output = deepsdf(xyz, latent)
            loss = criterion(output, sdf)
            loss.backward()
            optimizer.step()

            if i % 1000 == 0:
                print(loss.cpu().item())

    torch.save(deepsdf.state_dict(), 'test2.pth')