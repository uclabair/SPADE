import os
import glob
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm


def pull_embeds_and_stack(uni_root, image_id, save_root):
    all_embeds = sorted(glob.glob(os.path.join(uni_root, f'{image_id}/*.npy')))
    print(image_id, len(all_embeds))

    embed_stack = []
    for embed_p in all_embeds:
        embed = np.load(embed_p)
        embed_stack.append(embed)

    try:
        embed_stack = np.vstack(embed_stack)
        print(image_id, embed_stack.shape)
    except Exception as e:
        print(image_id, e)
        return 0
    np.save(os.path.join(save_root, f'{image_id}.npy'), embed_stack)

def main():
    uni_root = '/raid/PANDA_challenge/uni_embeds'
    save_root = '/raid/PANDA_challenge/uni_bags'
    labels_csv = pd.read_csv('/raid/PANDA_challenge/splits/labels.csv', index_col = 0)

    image_ids = labels_csv['image_id'].values

    for image_id in tqdm(image_ids):
        pull_embeds_and_stack(uni_root, image_id, save_root)



if __name__ == "__main__":
    main()