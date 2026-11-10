import torch
import torch.nn.functional as F


def response_logprobs(model, rollout, return_entropy: bool = False):
    """Log-prob of each response token given prompt + previous response tokens.
    Returns [B, Lr] (and per-token entropy [B, Lr] if requested). Gradients flow
    unless called under torch.no_grad(). Masked positions are not zeroed here."""
    device = next(model.parameters()).device
    q, qm = rollout.query_ids.to(device), rollout.query_mask.to(device)
    r, rm = rollout.response_ids.to(device), rollout.response_mask.to(device)
    ids = torch.cat([q, r], dim=1)
    attn = torch.cat([qm, rm], dim=1)
    pos = (attn.cumsum(-1) - 1).clamp(min=0)
    logits = model(input_ids=ids, attention_mask=attn, position_ids=pos).logits
    Lq, Lr = q.shape[1], r.shape[1]
    logits = logits[:, Lq - 1:Lq - 1 + Lr, :].float()
    logp_all = F.log_softmax(logits, dim=-1)
    logp = logp_all.gather(-1, r.unsqueeze(-1)).squeeze(-1)
    if return_entropy:
        # Entropy is only logged, never differentiated: compute it without
        # building a graph so no extra [B, Lr, V] tensors stay alive until backward.
        with torch.no_grad():
            lp = logp_all.detach()
            ent = -(lp.exp() * lp).sum(-1)
        return logp, ent
    return logp
