import torch
import torch.nn as nn

# --- Gradient Reversal Layer (as a Module) ---
class GradientReversalLayer(nn.Module):
    """
    Provides a clean interface to use gradient reversal in nn.Sequential blocks.
    """
    def __init__(self, lambda_adv=1.0):
        """_summary_

        Args:
            lambda_adv (float, optional): A scaling factor that controls how strongly the gradients are reversed.
             Defaults to 1.0.
        """
        super().__init__()
        self.lambda_adv = lambda_adv  # Can be modified dynamically
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_adv)

# --- Autograd Function for Reversal ---
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_adv):
        ctx.save_for_backward(torch.tensor(lambda_adv))
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        lambda_adv, = ctx.saved_tensors
        return grad_output.neg() * lambda_adv.item(), None

# --- Fair Hiring Model ---
class FairHiringModel(nn.Module):
    def __init__(self, input_dim, lambda_adv=1.0):
        super().__init__()
        self.lambda_adv = lambda_adv  # Exposed for easy modification
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Primary task: hiring prediction
        self.primary = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Adversary with built-in gradient reversal
        self.adversary = nn.Sequential(
            GradientReversalLayer(lambda_adv),  # Now part of the adversary
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        rep = self.shared(x)
        primary_pred = self.primary(rep)
        sensitive_pred = self.adversary(rep)  # GRL happens inside adversary
        return primary_pred, sensitive_pred

if __name__ == "__main__":
    # Test the model
    model = FairHiringModel(input_dim=20, lambda_adv=1.0)
    dummy_input = torch.randn(5, 20)
    primary_out, sensitive_out = model(dummy_input)
    print("Primary Output Shape:", primary_out.shape)  # (5, 1)
    print("Sensitive Output Shape:", sensitive_out.shape)  # (5, 1)