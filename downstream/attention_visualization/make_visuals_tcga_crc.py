import os
import numpy as np
import pandas as pd
import openslide
import glob
import pickle as pkl
from tqdm import tqdm
import argparse


from visualization_utils import *

def run_slide_attention_visual(locations_df, slide_path, tissue_mask, prob_type = 'uni', cancer_mask = None):
    locations = locations_df[['x', 'y']].values
    slide = openslide.OpenSlide(slide_path)
    dimensions = slide.level_dimensions
    tissue_pil = Image.open(tissue_mask)
    mask = np.array(Image.open(tissue_mask))
    
    thumb = slide.get_thumbnail(dimensions[-2])
    dimensions_mask = dimensions[-2][::-1]
    mask = resize(
        mask,
        dimensions_mask,
        mode = 'edge',
        anti_aliasing = False,
        anti_aliasing_sigma = None,
        preserve_range = True,
        order = 0)

    cancer_mask = resize(
            cancer_mask,
            dimensions_mask,
            mode = 'edge',
            anti_aliasing = False,
            anti_aliasing_sigma = None,
            preserve_range = True,
            order = 0
        )
    downsample_rate = slide.level_dimensions[0][0]/mask.shape[1]
    locations_downsample = (locations/downsample_rate).astype('int')
    if prob_type == 'uni':
        probs = locations_df['uni_attn'].values
    elif prob_type == 'spade':
        probs = locations_df['spade_attn'].values
        
    heatmap = tumor_heatmap(
                slide,
                mask,
                thumb,
                probs,
                locations_downsample,
                tile_size=224,
                mag=20,
                downsample_rate=downsample_rate,
                plot=False, percentile = True)
    
    heatmap = heatmap / np.max(heatmap)
    alpha = 0.7
    alpha_background = 0.95
    
    reduced_info = heatmap.copy() 
    reduced_info[(reduced_info <= 0.55)] = 0

    cmap = plt.get_cmap('coolwarm')
    color_block256 = (cmap(reduced_info) * 255)[:, :, :3].astype(np.uint8)

    thumbnail_np = np.array(thumb)
    thumbnail_resized = cv2.resize(thumbnail_np, (heatmap.shape[1], heatmap.shape[0]))

    background = np.zeros_like(thumbnail_resized)
    alpha_thumb = 0.8
    thumbnail_transparent = cv2.addWeighted(background, 1-alpha_thumb, thumbnail_resized, alpha_thumb, 0)
    
    overlay = cv2.addWeighted(thumbnail_resized, 1-alpha, color_block256, alpha, 0)
    overlay[reduced_info == 0] = thumbnail_resized[reduced_info == 0]  

    if cancer_mask is not None:
        mask = (cancer_mask * 255).astype(np.uint8) 
        edges = cv2.Canny(mask, threshold1=100, threshold2=200)
        edges_thick = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)  
        outline_overlay = overlay.copy()
        outline_overlay[edges_thick > 0] = [255, 255, 0]  
        return outline_overlay
    
    return overlay




def set_up_locs_and_imgs(test_df, case_list, args, attn_spade, attn_uni):
    all_locs = {}
    for case_id in tqdm(test_df['Case ID']):
        slide_set = case_list[case_id]['slide_names'].tolist()
        locs = []
        for slide_name in slide_set:
            files = sorted(glob.glob(os.path.join(args.feature_root, f'{slide_name}/*.npy')))
            for file in files:
                loc = file.split('/')[-1].split('.')[0]
                x = loc.split('_')[0]
                y = loc.split('_')[1]
                locs.append([case_id, slide_name, int(x), int(y)])
                
        locs = pd.DataFrame(locs, columns = ['Case ID', 'slide_name', 'x', 'y'])
        all_locs[case_id] = locs

    for ind, row in test_df.iterrows():
        curr_attn_spade = attn_spade[ind]
        curr_attn_uni = attn_uni[ind]
        
        curr_df = all_locs[row['Case ID']]
        curr_df['uni_attn'] = curr_attn_uni
        curr_df['spade_attn'] = curr_attn_spade
        all_locs[row['Case ID']] = curr_df

    img_dict = {}
    for case_id in tqdm(test_df['Case ID']):
        slide_set = case_list[case_id]['slide_names'].tolist()
        slide_paths = []
        for slide_name in slide_set:
            slide_path = sorted(glob.glob(os.path.join(args.slide_root_1, f'{slide_name}.svs')))
            if len(slide_path) == 0:
                slide_path = sorted(glob.glob(os.path.join(args.slide_root_2, f'{slide_name}.svs')))
            slide_path = slide_path[0]
            slide_paths.append(slide_path)
        img_dict[case_id] = slide_paths

    return all_locs, img_dict

def main(args):
    os.makedirs(args.save_root, exist_ok = True)
    attn_spade = np.load(args.attentions_spade, allow_pickle = True)
    attn_uni = np.load(args.attentions_uni, allow_pickle = True)

    test_df = pd.read_csv(args.test_dataframe, index_col = 0)

    with open(args.case_list_mapping, 'rb') as f:
        case_list = pkl.load(f)

    all_locs, img_dict = set_up_locs_and_imgs(test_df, case_list, args, attn_spade, attn_uni)
    all_case_ids = list(all_locs.keys())

    for case_id in tqdm(case_ids_to_run):
        locations_df = all_locs[case_id]
        slide_paths = img_dict[case_id]

        if len(slide_paths) > 1:
            print(case_id)
            continue
        else:
            slide_path = slide_paths[0]
            slide_name = slide_path.split('/')[-1].split('.')[0]
            tissue_mask_path = os.path.join(args.tissue_mask_root, f'{slide_name}.png')

            overlay_uni = run_slide_attention_visual(locations_df, slide_path, tissue_mask_path, prob_type = 'uni', cancer_mask = cancer_mask)
            fig, ax = plt.subplots(1, 1, figsize = (30, 30))
            im = ax.imshow(overlay_uni)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(args.save_root, f'{slide_name}_{case_id}_uni.png'), dpi = 300)
            plt.savefig(os.path.join(args.save_root, f'{slide_name}_{case_id}_uni.svg'), dpi = 300, format = 'svg')
            plt.clf()
            plt.close()

            overlay_spade = run_slide_attention_visual(locations_df, slide_path, tissue_mask_path, prob_type = 'spade', cancer_mask = cancer_mask)

            fig, ax = plt.subplots(1, 1, figsize = (30, 30))
            im = ax.imshow(overlay_spade)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(args.save_root, f'{slide_name}_{case_id}_spade.png'), dpi = 300)
            plt.savefig(os.path.join(args.save_root, f'{slide_name}_{case_id}_spade.svg'), dpi = 300, format = 'svg')
            plt.clf()
            plt.close()

            np.save(os.path.join(args.save_root, f'{slide_name}_{case_id}_spade.npy'), overlay_spade)
            np.save(os.path.join(args.save_root, f'{slide_name}_{case_id}_uni.npy'), overlay_uni)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--attentions_spade', type = str, default = ''
    )
    parser.add_argument(
        '--attentions_uni', type = str, default = ''
    )
    parser.add_argument(
        '--test_dataframe', type = str, default = ''
    )
    parser.add_argument(
        '--case_list_mapping', type = str, default = ''
    )
    parser.add_argument(
        '--slide_root_1', type = str, default = '/raid/TCGA/TCGA_ALL_CANCER/TCGA-COAD'
    )
    parser.add_argument(
        '--slide_root_2', type = str, default = '/raid/TCGA/TCGA_ALL_CANCER/TCGA-READ'
    )
    parser.add_argument(
        '--tissue_mask_root', type = str, default = '/raid/TCGA/tissue_masks'
    )
    parser.add_argument(
        '--save_root', type = str, default = ''
    )
    parser.add_argument(
        '--feature_root', type = str, default = '/raid/TCGA/TCGA_CRC_uni_224_20x'
    )
    args = parser.parse_args()
    main(args)