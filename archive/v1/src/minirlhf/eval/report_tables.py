"""Side-by-side sample table.

python -m minirlhf.eval.report_tables --config configs/default.yaml --tags base tuned --n 10
"""
from __future__ import annotations

from minirlhf.eval._util import base_parser, cfg_from_args, common_prompts, gen_path, out_dir, read_generations


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", "<br>")


def format_samples(tags, gens_by_tag, rewards_by_tag, n=10) -> str:
    """rewards_by_tag: {tag: {id: reward}}."""
    by_id = {t: {r["id"]: r for r in gens_by_tag[t]} for t in tags}
    ids = common_prompts({t: gens_by_tag[t] for t in tags})[:n]
    lines = ["| Prompt | " + " | ".join(f"{t} (reward)" for t in tags) + " |",
             "|" + "|".join(["---"] * (len(tags) + 1)) + "|"]
    for i in ids:
        cells = [_cell(by_id[tags[0]][i]["prompt"])]
        for t in tags:
            cells.append(f"{_cell(by_id[t][i]['response'])} **({rewards_by_tag[t][i]:+.2f})**")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def run(cfg, tags, n=10, device=None):
    from minirlhf.common.device import get_device
    from minirlhf.reward.factory import load_scorer
    scorer = load_scorer(cfg, device or get_device(cfg))
    gens = {t: read_generations(gen_path(cfg, t)) for t in tags}
    rewards = {t: dict(zip([g["id"] for g in gens[t]],
                           scorer.score([g["prompt"] for g in gens[t]], [g["response"] for g in gens[t]]).tolist()))
               for t in tags}
    md = f"# Sample responses (first {n} shared prompts)\n\n" + format_samples(tags, gens, rewards, n)
    path = out_dir(cfg) / "samples.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md)
    print(f"[report_tables] wrote {path}")
    return path


def main(argv=None):
    p = base_parser(__doc__)
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--n", type=int, default=10)
    a = p.parse_args(argv)
    run(cfg_from_args(a), a.tags, a.n)


if __name__ == "__main__":
    main()
