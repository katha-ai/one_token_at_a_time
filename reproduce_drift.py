import torch
import torch.nn.functional as F

def test_drift():
    # Simulate a typical attention logit matrix in bfloat16
    # Shape: (batch=1, heads=1, q_len=32, k_len=32)
    torch.manual_seed(42)
    logits = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16, device='cpu')
    
    # Baseline: standard softmax
    probs_orig = F.softmax(logits, dim=-1)
    
    # "Boosting" with factor 1.0 using the current logic
    boost_factor = 1.0
    probs_boosted = probs_orig.clone()
    # Apply boost (no-op for 1.0)
    probs_boosted[..., -1:, 10:20] *= boost_factor
    
    # Renormalize ENTIRE matrix
    sum_probs = probs_boosted.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    probs_final = probs_boosted / sum_probs
    
    # Check difference
    diff = (probs_orig - probs_final).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"Max diff with factor 1.0: {max_diff}")
    print(f"Mean diff with factor 1.0: {mean_diff}")
    
    # Check if history (non-last rows) changed
    history_diff = diff[..., :-1, :].max().item()
    print(f"Max history diff (rows 0-30): {history_diff}")

    # Check tokens argmax drift simulation
    # Let's say we have some output projection
    proj = torch.randn(32, 100, dtype=torch.bfloat16, device='cpu')
    # Hidden states from attention (simplified)
    h_orig = torch.matmul(probs_orig[..., -1, :], torch.randn(32, 32, dtype=torch.bfloat16, device='cpu'))
    h_final = torch.matmul(probs_final[..., -1, :], torch.randn(32, 32, dtype=torch.bfloat16, device='cpu'))
    
    # Logits for next token
    l_orig = torch.matmul(h_orig, proj)
    l_final = torch.matmul(h_final, proj)
    
    argmax_orig = l_orig.argmax().item()
    argmax_final = l_final.argmax().item()
    
    if argmax_orig != argmax_final:
        print(f"Divergence detected! Argmax {argmax_orig} vs {argmax_final}")
    else:
        print("No argmax divergence in this single-step trial (but drift exists).")

if __name__ == "__main__":
    test_drift()
