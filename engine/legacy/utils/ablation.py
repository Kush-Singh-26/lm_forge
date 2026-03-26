"""
engine/utils/ablation.py

Ablation runner — run multiple config variants from a single YAML spec.

Design:
  One base config + a list of overrides = N experiments run sequentially
  (or in parallel if you have multiple GPUs).

  Each variant gets its own:
    • experiment name  (base_name + variant suffix)
    • output directory
    • HF Hub repo     (base_repo + "-" + suffix)
    • results entry in ablation_results.json

Ablation YAML format::

    base: experiments/exp_001_gqa_rope/config.yaml   # base config to inherit

    sweep:
      - name: "rope_gqa"
        model:
          attention: {type: gqa, num_kv_heads: 2}
          positional: {type: rope}

      - name: "rope_mha"
        model:
          attention: {type: mha}
          positional: {type: rope}

      - name: "alibi_gqa"
        model:
          attention: {type: gqa, num_kv_heads: 2}
          positional: {type: alibi}

      - name: "nope_gqa"
        model:
          attention: {type: gqa, num_kv_heads: 2}
          positional: {type: none}

Usage::

    from engine.utils import AblationRunner

    runner = AblationRunner("experiments/ablation_001/ablation.yaml")
    runner.run(train_fn)        # train_fn(exp_config) → final eval loss

    # Inspect results
    results = runner.load_results()
    runner.print_table(results)
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

from engine.config.schema import ExperimentConfig, load_experiment_config, _merge


class AblationRunner:
    """
    Runs a sweep of config variants and collects results.

    Args:
        ablation_yaml : Path to the ablation spec YAML.
        results_path  : Where to write ablation_results.json.
                        Defaults to same directory as ablation_yaml.
    """

    def __init__(
        self,
        ablation_yaml: str | Path,
        results_path: Optional[str | Path] = None,
    ) -> None:
        self.ablation_path = Path(ablation_yaml)
        spec = yaml.safe_load(self.ablation_path.read_text())
        self.base_config_path = Path(spec["base"])
        self.sweep: List[Dict[str, Any]] = spec.get("sweep", [])

        if results_path is None:
            self.results_path = self.ablation_path.parent / "ablation_results.json"
        else:
            self.results_path = Path(results_path)

    def _build_variant(self, variant_spec: Dict[str, Any]) -> ExperimentConfig:
        """Load base config, apply variant overrides, return ExperimentConfig."""
        exp = load_experiment_config(self.base_config_path)
        name = variant_spec.get("name", f"variant_{id(variant_spec)}")
        exp.name = name

        # Apply per-variant overrides to model and training sections
        if "model" in variant_spec:
            _merge(exp.model, variant_spec["model"])
        if "training" in variant_spec:
            _merge(exp.training, variant_spec["training"])

        # Override hub repo suffix so variants land in separate Hub repos
        if exp.hub.repo_id:
            base_repo = exp.hub.repo_id.rstrip("/")
            exp.hub.repo_id = f"{base_repo}-{name}"

        return exp

    def run(
        self,
        train_fn: Callable[[ExperimentConfig], float],
        skip_existing: bool = True,
    ) -> List[Dict]:
        """
        Run all sweep variants.

        Args:
            train_fn      : A function that takes an ExperimentConfig, trains
                            the model, and returns the final eval loss (float).
            skip_existing : If True, skip variants that already have results
                            in ablation_results.json.

        Returns:
            List of result dicts (also written to ablation_results.json).
        """
        existing = self._load_existing()

        print(f"\n{'━'*60}")
        print(f"  Ablation: {self.ablation_path.name}")
        print(f"  Variants: {len(self.sweep)}")
        if skip_existing and existing:
            print(f"  Skipping: {len(existing)} already done")
        print(f"{'━'*60}\n")

        for i, variant_spec in enumerate(self.sweep):
            name = variant_spec.get("name", f"variant_{i}")

            if skip_existing and name in {r["name"] for r in existing}:
                print(f"[{i+1}/{len(self.sweep)}] Skipping '{name}' (already done)")
                continue

            print(f"\n[{i+1}/{len(self.sweep)}] Running variant: {name}")
            exp = self._build_variant(variant_spec)

            t0 = time.perf_counter()
            eval_loss = None
            error = None

            try:
                eval_loss = float(train_fn(exp))
            except Exception as e:
                error = str(e)
                print(f"  ✗ Failed: {e}")

            elapsed = time.perf_counter() - t0

            result = {
                "name":      name,
                "eval_loss": eval_loss,
                "time_min":  round(elapsed / 60, 2),
                "error":     error,
                "config": {
                    "attention_type":  exp.model.attention.type,
                    "num_kv_heads":    exp.model.attention.num_kv_heads,
                    "positional_type": exp.model.positional.type,
                    "ffn_type":        exp.model.ffn.type,
                    "norm_type":       exp.model.norm.type,
                    "max_steps":       exp.training.max_steps,
                    "lr":              exp.training.lr,
                },
            }

            if eval_loss is not None:
                print(f"  ✓  eval_loss={eval_loss:.4f}  time={elapsed/60:.1f}min")

            existing.append(result)
            self._write_results(existing)

        self.print_table(existing)
        return existing

    # ── Result handling ───────────────────────────────────────────────────

    def _load_existing(self) -> List[Dict]:
        if self.results_path.exists():
            return json.loads(self.results_path.read_text())
        return []

    def _write_results(self, results: List[Dict]) -> None:
        self.results_path.write_text(json.dumps(results, indent=2))

    def load_results(self) -> List[Dict]:
        """Load and return existing results."""
        return self._load_existing()

    def print_table(self, results: Optional[List[Dict]] = None) -> None:
        """Print a sorted results table to stdout."""
        if results is None:
            results = self._load_existing()
        if not results:
            print("No results yet.")
            return

        # Sort by eval_loss (None/failed at the bottom)
        ok  = [r for r in results if r.get("eval_loss") is not None]
        bad = [r for r in results if r.get("eval_loss") is None]
        ok  = sorted(ok, key=lambda r: r["eval_loss"])

        header = f"{'Rank':<5} {'Name':<25} {'Loss':>8} {'Attn':>10} {'PE':>10} {'FFN':>8} {'Time(m)':>8}"
        print(f"\n{'━'*len(header)}")
        print(header)
        print(f"{'━'*len(header)}")

        for rank, r in enumerate(ok, 1):
            cfg = r.get("config", {})
            attn_str = f"{cfg.get('attention_type','?')}({cfg.get('num_kv_heads','?')}kv)"
            print(
                f"{rank:<5} {r['name']:<25} {r['eval_loss']:>8.4f}"
                f" {attn_str:>10} {cfg.get('positional_type','?'):>10}"
                f" {cfg.get('ffn_type','?'):>8} {r.get('time_min','?'):>8}"
            )
        for r in bad:
            print(f"{'✗':<5} {r['name']:<25} {'FAILED':>8}  {r.get('error','')[:40]}")
        print(f"{'━'*len(header)}\n")
