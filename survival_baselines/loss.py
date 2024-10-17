import torch
import numpy as np

def custom_neg_partial_log_likelihood(log_hz, event, time):
    # Ensure inputs are 1D
    log_hz = log_hz.view(-1)
    event = event.view(-1)
    time = time.view(-1)
    
    # Sort by time
    _, sort_idx = torch.sort(time, descending=True)
    log_hz_sorted = log_hz[sort_idx]
    event_sorted = event[sort_idx]
    
    # Calculate the cumulative sum of exponentiated log hazards
    epsilon = 1e-7
    cumsum_hz = torch.cumsum(torch.exp(log_hz_sorted), dim=0) + epsilon
    log_cumsum_hz = torch.log(cumsum_hz)
    
    # Select events
    event_idx = (event_sorted == 1)
    
    if event_idx.sum() == 0:
        return torch.tensor(0.0, requires_grad=True)
    
    # Calculate partial likelihood
    partial_likelihood = log_hz_sorted[event_idx] - log_cumsum_hz[event_idx]
    partial_likelihood = torch.clamp(partial_likelihood, min=-1e3, max=1e3) # clip to prevent VERY small values
    
    # Return negative partial likelihood
    return -partial_likelihood.mean()