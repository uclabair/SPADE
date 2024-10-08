import torch

def debug_tensor(tensor, name, step, threshold=1e6):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"Step {step}: NaN or Inf detected in {name}")
        print(f"Shape: {tensor.shape}")
        print(f"NaN count: {torch.isnan(tensor).sum().item()}")
        print(f"Inf count: {torch.isinf(tensor).sum().item()}")
        return False
    if tensor.abs().max() > threshold:
        print(f"Step {step}: Large values detected in {name}")
        print(f"Max abs value: {tensor.abs().max().item()}")
    print(f"Step {step}: {name} - min: {tensor.min().item():.4f}, max: {tensor.max().item():.4f}, mean: {tensor.mean().item():.4f}, std: {tensor.std().item():.4f}")
    return True


def check_for_inf_nan_grads(model):
    """
    Check if any of the gradients in the model are inf or nan.
    
    Args:
    model (nn.Module): The PyTorch model to check.
    
    Returns:
    bool: True if any inf or nan gradients are found, False otherwise.
    """
    has_inf_nan = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                print(f"Inf or NaN gradient detected in {name}")
                print(f"Gradient stats for {name}:")
                print(f"  Shape: {param.grad.shape}")
                print(f"  Min: {param.grad.min().item()}")
                print(f"  Max: {param.grad.max().item()}")
                print(f"  Mean: {param.grad.mean().item()}")
                print(f"  Std: {param.grad.std().item()}")
                has_inf_nan = True
    
    #if has_inf_nan:
    #    print("Inf or NaN gradients detected in the model.")
    #else:
    #    print("No Inf or NaN gradients detected in the model.")
    
    return has_inf_nan