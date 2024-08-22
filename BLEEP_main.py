import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import config as CFG
from models import CLIPModel, ZINBencoder
from utils import AvgMeter
from PIL import Image
import torchvision.transforms.functional as TF
import random
from dataset import CLIPDataset


# Load data
folds = np.load(CFG.folds_path, allow_pickle=True)
df1 = pd.read_csv(CFG.coords_data_path_v1)
df2 = pd.read_csv(CFG.coords_data_path_v2)
df = pd.merge(df1, df2, on=['id', 'patch_x', 'patch_y'], how='inner')

train_df = df[df['id'].isin(list(folds[0][0]))]
val_df = df[df['id'].isin(list(folds[0][1]))]

train_dataset = CLIPDataset(train_df, CFG)
val_dataset = CLIPDataset(val_df, CFG)

train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers)

loss_meter = AvgMeter()

model = CLIPModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)

for ep in range(CFG.num_epochs):
    model.train()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for i, batch in enumerate(tqdm_object):
        batch = {k: v.cuda() for k, v in batch.items() if k in ["image_features", "reduced_expression"]}

        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        count = batch["image_features"].size(0)
        loss_meter.update(loss.item(), count)
        tqdm_object.set_postfix(train_loss=loss_meter.avg)

    checkpoint_path = os.path.join(CFG.checkpoint_dir, f"0821_upd_ep{ep}_loss{loss_meter.avg:.6f}.pt")
    torch.save(model.state_dict(), checkpoint_path)

# import os
# import numpy as np
# from tqdm import tqdm
# import scipy.io as sio
# import cv2
# import torch
# from torch import nn
# import torch.distributed as dist
# import torch.utils.data.distributed
# import pandas as pd
# import config as CFG
# # from dataset import CLIPDataset
# from models import CLIPModel, CLIPModel_ViT, CLIPModel_ViT_L, CLIPModel_CLIP, CLIPModel_resnet101, CLIPModel_resnet152, ZINBencoder
# from utils import AvgMeter
# from torch.utils.data import DataLoader

# import scanpy as sc
# import argparse


# import os
# import cv2
# import pandas as pd
# import torch
# from sklearn.decomposition import TruncatedSVD
# # from scipy.sparse import csr_matrix
# import numpy as np
# import torchvision.transforms.functional as TF
# import random
# from PIL import Image

# class CLIPDataset(torch.utils.data.Dataset):
#     def __init__(self, df):
#         self.df = df

#     def transform(self, image):
#         image = Image.fromarray(image)
#         # Random flipping and rotations
#         if random.random() > 0.5:
#             image = TF.hflip(image)
#         if random.random() > 0.5:
#             image = TF.vflip(image)
#         angle = random.choice([180, 90, 0, -90])
#         image = TF.rotate(image, angle)
#         return np.asarray(image)

#     def __getitem__(self, idx):
#         # try:
#         row = self.df.iloc[idx]
#         id_ = row['id']
#         patch_x = row['patch_x']
#         patch_y = row['patch_y']

#         index = self.df.iloc[idx]['bead']
        
#         reduced_matrix = np.load("/raid/HEST-1K/071524_hvg_matrix/hvg_matrix_{0}.npy".format(id_)).T
#         # print(idx, id_)
#         image_features = np.load('/raid/mpleasure/data_deepstorage/st_projects/visium/uni_embeds_224_20x_normalized_no_thresh_08_14/{0}/{1}_{2}.npy'.format(id_, patch_x, patch_y))
#         # print(idx, id_,image_features.shape)
#         item = {}
#         item['image_features'] = torch.tensor(image_features).float() 
#         item['reduced_expression'] = torch.tensor(reduced_matrix[index]).float()  
#         # print(idx, len(np.unique(reduced_matrix[index])))
#         # item['barcode'] = barcode
#         item['spatial_coords'] = [patch_x,patch_y]

#         return item


#     def __len__(self):
#         return self.df.shape[0]


# folds = np.load('/raid/eredekop/071024_ST/data/folds.npy', allow_pickle=True)
# # df2 = pd.read_csv('/raid/mpleasure/data_deepstorage/st_projects/visium/all_coords_visium_data_v2.csv')
# # df1 = pd.read_csv('/raid/mpleasure/data_deepstorage/st_projects/visium/all_coords_visium_data.csv')
# # df = pd.merge(df1, df2, on=['id', 'patch_x', 'patch_y'], how='inner')
# df2 = pd.read_csv('/raid/mpleasure/data_deepstorage/st_projects/visium/all_coords_visium_data_v3.csv')
# df1 = pd.read_csv('/raid/mpleasure/data_deepstorage/st_projects/visium/all_coords_visium_data.csv')
# df = pd.merge(df1, df2, on=['id', 'patch_x', 'patch_y'], how='inner')

# train_df = df[df['id'].isin(list(folds[0][0]))]
# val_df = df[df['id'].isin(list(folds[0][1]))]

# train_dataset = CLIPDataset(train_df)
# val_dataset = CLIPDataset(val_df)


# train_sampler = None
# train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=8, sampler=train_sampler)

# loss_meter = AvgMeter()


# # enc = ZINBencoder().cuda()
# # enc.load_state_dict(torch.load('/raid/eredekop/071024_ST/checkpoints/tmp_zinb_enc_ep0_loss_0.4284879542529419.pt'))

# model = CLIPModel().cuda()
# optimizer = torch.optim.AdamW(
#     model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
# )

# # tqdm_object = tqdm(train_loader, total=len(train_loader))

# for ep in range(100):
#     model.train()
#     tqdm_object = tqdm(train_loader, total=len(train_loader))
#     for i, batch in enumerate(tqdm_object):
#         # try:
#             # if i != 835:
#                 # print(batch['image_features'].shape)
#                 batch = {k: v.cuda() for k, v in batch.items() if k == "image_features" or k == "reduced_expression"}
#                 # print(batch['image_features'].shape, batch['reduced_expression'].shape)
#                 loss = model(batch)
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()

#                 count = batch["image_features"].size(0)
#                 loss_meter.update(loss.item(), count)

#                 tqdm_object.set_postfix(train_loss=loss_meter.avg)#, lr=get_lr(optimizer))


        
#         # except:
#         #     pass
#     torch.save(model.state_dict(), "/raid/eredekop/071024_ST/checkpoints/0816_upd_ep{0}_loss{1}.pt".format(ep,loss_meter.avg))
