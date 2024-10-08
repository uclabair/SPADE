import torch
from torch import nn
import torch.nn.functional as F

import config as CFG
from modules import ProjectionHead
# from HIPT_4K.hipt_4k import HIPT_4K
# from HIPT_4K.hipt_model_utils import get_vit256, get_vit4k, eval_transforms
# from HIPT_4K.hipt_heatmap_utils import *
import torch
from torch import nn
import torch.nn.functional as F

#2265x256, 2277x256
def find_matches(spot_embeddings, query_embeddings, top_k=1):
    #find the closest matches 
    spot_embeddings = torch.tensor(spot_embeddings)
    query_embeddings = torch.tensor(query_embeddings)
    query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
    spot_embeddings = F.normalize(spot_embeddings, p=2, dim=-1)
    dot_similarity = query_embeddings @ spot_embeddings.T   #2277x2265
    print(dot_similarity.shape)
    _, indices = torch.topk(dot_similarity.squeeze(0), k=top_k)
    
    return indices.cpu().numpy()

import torch
import torch.nn as nn

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

    
from torch.nn import Parameter
def buildNetwork(layers, type, activation="relu"):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
    return nn.Sequential(*net)



class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=CFG.image_embedding,
        spot_embedding=CFG.spot_embedding,
        spot_encoder=None,
    ):
        super().__init__()
        # self.image_encoder = ImageEncoder()
        # self.spot_encoder = spot_encoder
        self.image_projection = ProjectionHead(embedding_dim=image_embedding) #aka the input dim, 2048 for resnet50
        self.spot_projection = ProjectionHead(embedding_dim=spot_embedding) #3467 shared hvgs
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and spot Features
        image_features = batch["image_features"]
        spot_features = batch["reduced_expression"]
        # spot_features = self.spot_encoder(batch)
        
        # Getting Image and Spot Embeddings (with same dimension) 
        image_embeddings = self.image_projection(image_features)
        spot_embeddings = self.spot_projection(spot_features)

        # Calculating the Loss
        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity = spot_embeddings @ spot_embeddings.T
        targets = F.softmax(
            (images_similarity + spots_similarity) / 2 * self.temperature, dim=-1
        )
        spots_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + spots_loss) / 2.0 # shape: (batch_size)
        return loss.mean()
    
class CLIPModel_reconstr(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=CFG.image_embedding,
        spot_embedding=CFG.spot_embedding,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder()
#         self.spot_encoder = SpotEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding) #aka the input dim, 2048 for resnet50
        self.spot_projection = ProjectionHead(embedding_dim=spot_embedding) #3467 shared hvgs
        self.temperature = temperature
        
        ### for the reconstruction
        encodeLayer = [256, 64]
        decodeLayer = [64, 256]
        input_dim = 7968
        z_dim = 512
        self.encoder = buildNetwork([input_dim]+encodeLayer, type="encode", activation="relu")
        self.decoder = buildNetwork([z_dim]+decodeLayer, type="decode", activation="relu")
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)
        self._dec_mean = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), MeanAct())
        self._dec_disp = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), DispAct())
        self._dec_pi = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), nn.Sigmoid())

#         self.mu = Parameter(torch.Tensor(n_clusters, z_dim))
        self.zinb_loss = ZINBLoss().cuda()
        self.sigma = 2.5
        
    def soft_assign(self, z):
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.mu)**2, dim=2) / self.alpha)
        q = q**((self.alpha+1.0)/2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()
        return q
    def forward(self, batch):
        # Getting Image and spot Features
        image_features = batch["image_features"] #self.image_encoder()
        spot_features = batch["reduced_expression"]
#         spot_features = self.spot_encoder(batch["reduced_expression"])
        
        # Getting Image and Spot Embeddings (with same dimension) 
        image_embeddings = self.image_projection(image_features)
        spot_embeddings = self.spot_projection(spot_features)

        # Calculating the Loss
        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity = spot_embeddings @ spot_embeddings.T
        targets = F.softmax(
            (images_similarity + spots_similarity) / 2 * self.temperature, dim=-1
        )
        
        
        spots_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        
        ## for the reconstruction
        h = self.encoder(spot_features+torch.randn_like(spot_features) * self.sigma)
        z = self._enc_mu(h)
        h = self.decoder(z)
        _mean = self._dec_mean(h)
        _disp = self._dec_disp(h)
        _pi = self._dec_pi(h)

        h0 = self.encoder(spot_features)
        z0 = self._enc_mu(h0)
#         q = self.soft_assign(z0)
        zinb_loss = self.zinb_loss(x=spot_features, mean=_mean, disp=_disp, pi=_pi)
        
        
        loss =  (images_loss + spots_loss + zinb_loss) / 3.0 # shape: (batch_size)
        
      
        return loss.mean()







def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()



from torch.nn import Parameter
def buildNetwork(layers, type, activation="relu"):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
    return nn.Sequential(*net)


class AttentionPoolingMod(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionPoolingMod, self).__init__()

        self.W_q = nn.Linear(input_dim, hidden_dim)
        self.W_k = nn.Linear(input_dim, hidden_dim)
        self.W_v = nn.Linear(input_dim, hidden_dim)

        self.softmax = nn.Softmax(dim = -1)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, center_embedding, neighbor_embedding, mask = None):
        query = self.W_q(center_embedding) # (B, D) - batch size by dimension

        keys = self.W_k(neighbor_embedding) # (B, N, D) - batch, number of neighbors, dimension
        values = self.W_v(neighbor_embedding) # (B, N, D)

        scores = torch.matmul(query.unsqueeze(1), keys.transpose(1, 2)).squeeze(1) # get out (B, N)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        weight = self.softmax(scores) # (B, N)

        agg = torch.bmm(weight.unsqueeze(1), values).squeeze(1) # (B, D)
        updated_center = self.layer_norm(query + agg)

        return updated_center, weight


class CLIPModel_Neighbor(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=CFG.image_embedding,
        spot_embedding=CFG.spot_embedding,
        projection_dim = None,
        args=CFG,
    ):
        super().__init__()
        self.projection_dim = projection_dim
        self.image_projection = ProjectionHead(embedding_dim=image_embedding, projection_dim=self.projection_dim)
        self.spot_projection = ProjectionHead(embedding_dim=spot_embedding, projection_dim=self.projection_dim)
        self.temperature = temperature

        self.image_attention = AttentionPoolingMod(input_dim = image_embedding, hidden_dim = self.projection_dim)
        self.gene_attention = AttentionPoolingMod(input_dim = spot_embedding, hidden_dim = self.projection_dim)
        self.args = args

        self.concat_img = nn.Linear(projection_dim * 2, projection_dim)

    def forward(self, batch):
        # Getting Image and spot Features
        image_features = batch["image_features"]
        spot_features = batch["reduced_expression"]
        neighbor_image_feats = batch['image_neighbors']
        neighbor_gene_feats = batch['gene_neighbors']
        img_mask = batch['image_mask']
        gene_mask = batch['gene_mask']
        # spot_features = self.spot_encoder(batch)
        
        # Getting Image and Spot Embeddings (with same dimension) 
        image_embeddings = self.image_projection(image_features)
        spot_embeddings = self.spot_projection(spot_features)

        image_attn, image_attn_weights = self.image_attention(image_features, neighbor_image_feats, mask = img_mask) # B, projection_dim
        #gene_attn, gene_attn_weights = self.gene_attention(spot_features, neighbor_image_feats, mask = img_mask) # B, projection_dim

        #spot_embeddings = torch.cat([spot_embeddings, gene_attn], dim = -1) # B, projection_dim * 2
        #spot_embeddings = self.concat_img(spot_embeddings)# B, projection_dim
        image_embeddings = torch.cat([image_embeddings, image_attn], dim = -1) # B, projection_dim * 2
        image_embeddings = self.concat_img(image_embeddings)# B, projection_dim


        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity = spot_embeddings @ spot_embeddings.T
        targets = F.softmax(
            (images_similarity + spots_similarity) / 2 * self.temperature, dim=-1
        )
        spots_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + spots_loss) / 2.0 # shape: (batch_size)

        
        return loss.mean()
    

class CLIPModelAttn(nn.Module):
    def __init__(
        self,
        image_embedding,
        spot_embedding,
        projection_dim = 256,
        args = None
    ):
        super().__init__()
        # self.image_encoder = ImageEncoder()
        # self.spot_encoder = spot_encoder
        self.image_projection = ProjectionHead(embedding_dim=image_embedding, projection_dim=projection_dim) #aka the input dim, 2048 for resnet50
        self.spot_projection = ProjectionHead(embedding_dim=spot_embedding, projection_dim=projection_dim) #3467 shared hvgs
        self.temperature = args.temperature
        self.args = args

        self.image_attn = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.Tanh(),
            nn.Linear(projection_dim, 1) # 1 attention weight per neighbor
        )
        self.gene_attn = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.Tanh(),
            nn.Linear(projection_dim, 1) # 1 attention weight per neighbor
        )

        self.concat_img = nn.Linear(projection_dim * 2, projection_dim)
        self.concat_gene = nn.Linear(projection_dim * 2, projection_dim)


    def forward(self, batch):
        # Getting Image and spot Features
        image_features = batch["image_features"]
        spot_features = batch["reduced_expression"]
        neighbor_image_feats = batch['image_neighbors']
        neighbor_gene_feats = batch['gene_neighbors']
        img_mask = batch['image_mask']
        gene_mask = batch['gene_mask']
        # spot_features = self.spot_encoder(batch)
        
        # Getting Image and Spot Embeddings (with same dimension) 
        image_embeddings = self.image_projection(image_features) # B , projection_dim
        spot_embeddings = self.spot_projection(spot_features) # B , projection_dim
        neighbor_image_proj = self.image_projection(neighbor_image_feats) # B, n_neighbors, projection_dim
        neighbor_gene_proj = self.spot_projection(neighbor_gene_feats) # B, n_neighbors, projection_dim

        img_attn_scores = self.image_attn(neighbor_image_proj).squeeze(-1) # B, n_neighbors
        img_attn_scores = img_attn_scores.masked_fill(img_mask == 0, float('-inf')) # mask out padding - after softmax this will be 0
        img_attn_weights = F.softmax(img_attn_scores, dim = -1) # b, n_neighbors

        spot_attn_scores = self.gene_attn(neighbor_image_proj).squeeze(-1)
        spot_attn_scores = spot_attn_scores.masked_fill(gene_mask == 0, float('-inf'))
        spot_attn_weights = F.softmax(spot_attn_scores, dim = -1)

        # pool image weights and gene weights
        pooled_img_feats = torch.bmm(img_attn_weights.unsqueeze(1), neighbor_image_proj).squeeze(1) # B, projection_dim
        pooled_gene_feats = torch.bmm(spot_attn_weights.unsqueeze(1), neighbor_gene_proj).squeeze(1) # B, projection_dim

        # add to center
        if self.args.add_center:
            image_embed_attn = torch.cat([image_embeddings, pooled_img_feats], dim = -1) # B, projection_dim*2
            gene_embed_attn = torch.cat([spot_embeddings, pooled_gene_feats], dim = -1) # B, projection_dim*2

            # linear back to projection_dim
            image_embed_attn = self.concat_img(image_embed_attn)
            gene_embed_attn = self.concat_gene(gene_embed_attn)
        else:
            gene_embed_attn = pooled_gene_feats
            image_embed_attn = pooled_img_feats

        # Calculating the Loss
        logits = (spot_embeddings @ image_embed_attn.T) / self.temperature
        images_similarity = image_embed_attn @ image_embed_attn.T
        spots_similarity = spot_embeddings @ spot_embeddings.T
        targets = F.softmax(
            (images_similarity + spots_similarity) / 2 * self.temperature, dim=-1
        )
        spots_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + spots_loss) / 2.0 # shape: (batch_size)
        return loss.mean()