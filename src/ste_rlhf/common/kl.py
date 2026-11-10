import torch


def per_token_kl(logp, logp_ref, mask):
    """Per-token estimate of KL(pi || pi_ref) on tokens sampled from pi, shape [B, Lr].

    Uses the "k3" estimator: with d = logp_ref - logp, k3 = exp(d) - 1 - d.
    It is unbiased for KL(pi || pi_ref) when the tokens are sampled from pi itself,
    never negative (exp(d) >= 1 + d), and has much lower variance than the plain
    k1 = logp - logp_ref, which can come out negative on a batch (v2 pilot saw
    about -1 over ~200 tokens). Masked positions are 0.
    """
    d = (logp_ref - logp).clamp(-20.0, 20.0)  # guard exp() against overflow on outliers
    return (torch.exp(d) - 1.0 - d) * mask.to(logp.dtype)
