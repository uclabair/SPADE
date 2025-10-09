import torch
from torch import nn
import torch.nn.functional as F
import config as CFG
from modules import ProjectionHead
import torch
from torch import nn
import torch.nn.functional as F


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=CFG.image_embedding,
        spot_embedding=CFG.spot_embedding,
        spot_encoder=None,
    ):
        super().__init__()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding) #aka the input dim, 2048 for resnet50
        self.spot_projection = ProjectionHead(embedding_dim=spot_embedding) #3467 shared hvgs
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and spot Features
        image_features = batch["image_features"]
        spot_features = batch["reduced_expression"]

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
