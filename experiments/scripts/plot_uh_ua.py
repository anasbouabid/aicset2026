#!/usr/bin/env python3
"""Write figures/uh_ua.tex from archived T1 JSON. Does not overwrite results."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "results"
OUT = ROOT / "figures" / "uh_ua.tex"

XMAX = 0.85
YMIN, YMAX = 0.10, 1.00
W, H = 7.35, 5.55


def load(name: str) -> tuple[float, float]:
    payload = json.loads((RESULTS / name).read_text())
    uh = float(payload["honest_utility"]["pr_auc"])
    ua = float(payload["adversarial_utility"]["surrogate_label_agreement"])
    return uh, ua


def xy(uh: float, ua: float) -> tuple[float, float]:
    return W * uh / XMAX, H * (ua - YMIN) / (YMAX - YMIN)


def v5(stem: str, seed: int) -> str:
    if seed == 0:
        return f"paysim_natural_ctgan_r_v5_{stem}.json"
    return f"paysim_natural_ctgan_r_v5_seed{seed}_{stem}.json"


def v4_topk(seed: int) -> str:
    if seed == 0:
        return "paysim_natural_ctgan_r_v4_r05_typetopk.json"
    return f"paysim_natural_ctgan_r_v4_seed{seed}_r05_typetopk.json"


def ieee_rateonly(seed: int) -> str:
    if seed == 0:
        return "ieee_cis_natural_ctgan_r05_rateonly.json"
    return f"ieee_cis_natural_ctgan_r05_rateonly_seed{seed}.json"


SERIES = [
    (
        "unfilt",
        "PaySim unfiltered",
        [v5("unfiltered", s) for s in (0, 1, 2)],
        r"\draw[black] ({x:.3f},{y:.3f}) circle (1.7pt);",
    ),
    (
        "rateonly",
        r"PaySim $5\%$ (product)",
        [v5("r05_rateonly", s) for s in (0, 1, 2)],
        r"\fill[black] ({x:.3f},{y:.3f}) circle (1.7pt);",
    ),
    (
        "sanitize",
        "PaySim sanitizer",
        [
            v5(stem, s)
            for s in (0, 1, 2)
            for stem in ("r05_jitter01", "r05_jitter05", "r05_withhold")
        ],
        r"\node[inner sep=0, font=\scriptsize] at ({x:.3f},{y:.3f}) {{$\times$}};",
    ),
    (
        "topk",
        r"PaySim type top-$k$",
        [v4_topk(s) for s in (0, 1, 2)],
        r"\fill[black] ({x:.3f},{y:.3f}) -- ++(0,2.4pt) -- ++(2.1pt,-3.6pt) -- ++(-4.2pt,0) -- cycle;",
    ),
    (
        "ieee",
        r"IEEE-CIS $5\%$ CTGAN",
        [ieee_rateonly(s) for s in (0, 1, 2)],
        r"\draw[black] ({x:.3f},{y:.3f}) ++(-1.5pt,-1.5pt) rectangle ++(3.0pt,3.0pt);",
    ),
    (
        "dp",
        "PaySim DP floor",
        [
            "paysim_natural_dpind_eps1.json",
            "paysim_natural_dpind_eps10.json",
            "paysim_natural_dpclasscond_eps10.json",
            "paysim_natural_patectgan_eps10.json",
        ],
        r"\node[inner sep=0, font=\tiny, gray] at ({x:.3f},{y:.3f}) {{$+$}};",
    ),
]


def main() -> None:
    ticks_x = [0.0, 0.2, 0.4, 0.6, 0.8]
    ticks_y = [0.2, 0.4, 0.6, 0.8, 1.0]
    lines = [
        r"\begin{tikzpicture}[x=1cm,y=1cm]",
        rf"\draw[->] (0,0) -- ({W + 0.25:.2f},0);",
        rf"\node[font=\small, anchor=north] at ({W / 2:.2f},-0.72) {{$U_h$}};",
        rf"\draw[->] (0,0) -- (0,{H + 0.25:.2f}) node[above] {{$U_a$}};",
    ]
    for t in ticks_x:
        x = W * t / XMAX
        lines.append(rf"\draw[gray!25] ({x:.3f},0) -- ({x:.3f},{H:.3f});")
        lines.append(rf"\draw ({x:.3f},0) -- ++(0,-0.08) node[below,font=\scriptsize] {{{t:.1f}}};")
    for t in ticks_y:
        y = H * (t - YMIN) / (YMAX - YMIN)
        lines.append(rf"\draw[gray!25] (0,{y:.3f}) -- ({W:.3f},{y:.3f});")
        lines.append(rf"\draw (0,{y:.3f}) -- ++(-0.08,0) node[left,font=\scriptsize] {{{t:.1f}}};")

    csv_rows = ["series,file,uh,ua"]
    for key, _label, files, tmpl in SERIES:
        for name in files:
            uh, ua = load(name)
            csv_rows.append(f"{key},{name},{uh:.6f},{ua:.6f}")
            x, y = xy(uh, ua)
            lines.append("  " + tmpl.format(x=x, y=y))

    # One-column legend to the right of the axes (labels do not share a row).
    legend = [
        (r"\draw[black] ({x:.3f},{y:.3f}) circle (1.7pt);", "PaySim unfiltered"),
        (r"\fill[black] ({x:.3f},{y:.3f}) circle (1.7pt);", r"PaySim $5\%$ product"),
        (r"\node[inner sep=0, font=\scriptsize] at ({x:.3f},{y:.3f}) {{$\times$}};", "PaySim sanitizer"),
        (r"\fill[black] ({x:.3f},{y:.3f}) -- ++(0,2.4pt) -- ++(2.1pt,-3.6pt) -- ++(-4.2pt,0) -- cycle;", r"PaySim type top-$k$"),
        (r"\draw[black] ({x:.3f},{y:.3f}) ++(-1.5pt,-1.5pt) rectangle ++(3.0pt,3.0pt);", r"IEEE-CIS $5\%$"),
        (r"\node[inner sep=0, font=\scriptsize] at ({x:.3f},{y:.3f}) {{$+$}};", "PaySim DP floor"),
    ]
    lx = W + 0.55
    ly = H - 0.15
    step = 0.52
    for i, (mark, text) in enumerate(legend):
        y = ly - i * step
        lines.append("  " + mark.format(x=lx, y=y))
        lines.append(
            rf"  \node[anchor=west, font=\scriptsize] at ({lx + 0.48:.2f},{y:.2f}) {{{text}}};"
        )

    lines.append(r"\end{tikzpicture}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    (OUT.with_suffix(".csv")).write_text("\n".join(csv_rows) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(csv_rows) - 1} points)")


if __name__ == "__main__":
    main()
