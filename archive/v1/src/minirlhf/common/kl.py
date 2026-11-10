def per_token_kl(logp, logp_ref, mask):
    """Per-token KL estimate (logp - logp_ref) * mask, shape [B, Lr]."""
    return (logp - logp_ref) * mask.to(logp.dtype)
