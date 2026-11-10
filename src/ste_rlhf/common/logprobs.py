import torch
import torch.nn.functional as F


def _chunk(model, q, qm, r, rm, return_entropy):
    ids = torch.cat([q, r], dim=1)
    attn = torch.cat([qm, rm], dim=1)
    pos = (attn.cumsum(-1) - 1).clamp(min=0)  # correct position ids under left padding
    logits = model(input_ids=ids, attention_mask=attn, position_ids=pos).logits
    Lq, Lr = q.shape[1], r.shape[1]
    logits = logits[:, Lq - 1:Lq - 1 + Lr, :].float()
    logp_all = F.log_softmax(logits, dim=-1)
    logp = logp_all.gather(-1, r.unsqueeze(-1)).squeeze(-1)
    if not return_entropy:
        return logp, None
    with torch.no_grad():  # entropy is only logged; keep it out of the autograd graph
        lp = logp_all.detach()
        ent = -(lp.exp() * lp).sum(-1)
    return logp, ent


def response_logprobs(model, rollout, return_entropy: bool = False, micro_batch=None):
    """Per-token log-prob of each response token: [B, Lr] (plus entropy [B, Lr]).
    Gradients flow unless called under torch.no_grad(). Masked positions are not
    zeroed here. micro_batch chunks the batch to cut peak memory."""
    device = next(model.parameters()).device
    q, qm = rollout.query_ids.to(device), rollout.query_mask.to(device)
    r, rm = rollout.response_ids.to(device), rollout.response_mask.to(device)
    B = q.shape[0]
    mb = int(micro_batch) if micro_batch else B
    lps, ents = [], []
    for i in range(0, B, mb):
        sl = slice(i, i + mb)
        lp, en = _chunk(model, q[sl], qm[sl], r[sl], rm[sl], return_entropy)
        lps.append(lp)
        ents.append(en)
    logp = torch.cat(lps, 0)
    if return_entropy:
        return logp, torch.cat(ents, 0)
    return logp
