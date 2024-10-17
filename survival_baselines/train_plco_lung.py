from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import pandas as pd
import pickle as pkl

from tqdm import tqdm
import argparse

from torchsurv.loss import cox
from torchsurv.metrics.cindex import ConcordanceIndex
from torchsurv.loss.cox import neg_partial_log_likelihood
from torchsurv.loss.weibull import neg_log_likelihood, log_hazard, survival_function
from torchsurv.metrics.brier_score import BrierScore
from torchsurv.metrics.auc import Auc
from torchsurv.loss.momentum import Momentum

from models import *
from dataset import *
from loss import *

np.random.seed(1234)
torch.manual_seed(123)

def set_up(args):
    labels = pd.read_csv(args.labels, index_col = 0)

    if args.split_type == 'pkl':
        with open(args.splits, 'rb') as f:
            splits = pkl.load(f)

        df_train = labels[labels['slide_name'].isin(splits['train'])]
        df_val = labels[labels['slide_name'].isin(splits['val'])]

    train_dataset = BCR_Dataset(
        args.bag_root, df_train)
    val_dataset = BCR_Dataset(
        args.bag_root, df_val)
    

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size = 1, shuffle = False, num_workers = 2)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size = 1, shuffle = False, num_workers = 2)
    

    return train_dataloader, val_dataloader


def eval_step(model, val_dataloader, use_momentum):
    log_hz = []
    events = []
    times = []
    total_loss =0.0
    num_batch = 0
    with torch.no_grad():
        for batch in val_dataloader:
            bag, name, event, time = batch
            bag = bag.type('torch.FloatTensor').squeeze(0).cuda()
            event = event.cuda()
            time = time.cuda()
            
            if use_momentum:
                output = model.online_network(bag)
            else:
                output = model(bag)

            loss = custom_neg_partial_log_likelihood(output, event, time)
            log_hz.append(output.detach().cpu().numpy())
            events.append(event.detach().cpu().numpy())
            times.append(time.detach().cpu().numpy())
            total_loss += loss.item()
            
            num_batch += 1

    avg_loss = total_loss/num_batch
    log_hz = np.concatenate(log_hz)
    events = np.concatenate(events)
    times = np.concatenate(times)

    log_hz = torch.tensor(log_hz)
    events = torch.tensor(events, dtype = torch.bool)
    times = torch.tensor(times)

    cox_cindex = ConcordanceIndex()
    cind = cox_cindex(log_hz, events, times)
    print(cind)
    print(log_hz.mean())

    return avg_loss, cind
        
def train(model, dataloader, optimizer, epochs, use_momentum, val_dataloader, save_root):
    best_cind = 0.0

    for epoch in range(epochs):
        epoch_loss = 0
        for batch in tqdm(dataloader):
            bag, name, event, time = batch
            bag = bag.type('torch.FloatTensor').squeeze(0).cuda()
            event = event.cuda()
            time = time.cuda()

            optimizer.zero_grad()

            try:
                # Forward pass through online network
                output = model(bag)

                # Compute loss
                loss = custom_neg_partial_log_likelihood(output, event, time)

                # Backward and optimize
                loss.backward()
                optimizer.step()

                # Update target network
                if use_momentum:
                    model.update_target_network()

                epoch_loss += loss.item()
            except RuntimeError as e:
                print(f"Error in batch: {e}")
                print(f"Bag shape: {bag.shape}, Event shape: {event.shape}, Time shape: {time.shape}")
                continue

            torch.cuda.empty_cache()
    
        val_loss, cind = eval_step(model, val_dataloader, use_momentum)

        print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(dataloader)}, val cindex: {cind}, val loss: {val_loss}")
        if cind > best_cind:
            torch.save({
                'model': model.state_dict(),
            }, os.path.join(save_root, f'{epoch}_best_{cind}.pt'))


def main(args):
    dataloader_train, dataloader_val = set_up(args)

    save_root = os.path.join(args.save_root, f'survival_bcr_momentum_{args.use_momentum}')
    os.makedirs(save_root, exist_ok=True)

    model = MIL_Attention_fc(size_arg = 'uni')
    model = model.cuda()
    
    if args.use_momentum:
        momentum_model = SimpleMomentum(model)
        momentum_model = momentum_model.cuda()
        optimizer = torch.optim.Adam(momentum_model.online_network.parameters(), lr = 1e-4, weight_decay = 1e-5)
        train(momentum_model, dataloader_train, optimizer, args.epochs, args.use_momentum, dataloader_val, save_root)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4, weight_decay = 1e-5)
        train(model, dataloader_train, optimizer, args.epochs, args.use_momentum, dataloader_val, save_root)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--bag_root', type = str, default = '/raid/mpleasure/data_deepstorage/st_projects/bcr_downstream/presaved_bags_'
    )
    parser.add_argument(
        '--embeds_root', type = str, default = '/raid/mpleasure/data_deepstorage/st_projects/bcr_downstream/uni_embeds'
    )
    parser.add_argument(
        '--labels', type = str, default = '/raid/mpleasure/PLCO/parsed_data/lung/lung_survival_task.csv'
    )
    parser.add_argument(
        '--splits', type = str, default = '/raid/mpleasure/data_deepstorage/st_projects/bcr_downstream/splits_10_03.pkl'
    )
    parser.add_argument(
        '--save_root', type = str, default = '/raid/mpleasure/PLCO/parsed_data/lung/survival_model'
    )
    parser.add_argument(
        '--split_type', type = str, default = 'pkl'
    )
    parser.add_argument(
        '--use_momentum', type = bool, default = True
    )
    parser.add_argument(
        '--epochs', type = int, default = 20
    )
    args = parser.parse_args()
    main(args)
