#!/usr/bin/env python3
"""Generate pilot report for WSD Gold Dataset.

This script analyzes the pilot labeling results and generates
a comprehensive report with statistics and cost projections.

Usage:
    uv run python scripts/generate_pilot_report.py
"""

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import typer

from eng_words.wsd_gold.aggregate import aggregate_labels, get_aggregation_stats
from eng_words.wsd_gold.models import GoldExample, ModelOutput

app = typer.Typer(help="Generate pilot report")


def load_examples(path: Path) -> list[GoldExample]:
    """Load examples from JSONL file."""
    examples = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            examples.append(GoldExample.from_dict(data))
    return examples


def load_labels(path: Path) -> dict[str, ModelOutput]:
    """Load labels from JSONL file."""
    labels = {}
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            example_id = data["example_id"]
            output = ModelOutput.from_dict(data["output"])
            labels[example_id] = output
    return labels


def calculate_percentile(values: list[float], percentile: int) -> float:
    """Calculate percentile of values."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * percentile / 100
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_vals) else f
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def generate_report(
    examples: list[GoldExample],
    labels_by_provider: dict[str, dict[str, ModelOutput]],
    output_path: Path,
) -> None:
    """Generate markdown report."""
    lines = []
    lines.append("# WSD Gold Dataset - Pilot Report")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("\n## Overview")
    lines.append(f"\n- **Total examples**: {len(examples)}")
    lines.append(f"- **Providers tested**: {', '.join(labels_by_provider.keys())}")

    # Provider statistics
    lines.append("\n## Provider Statistics\n")
    lines.append("| Provider | Labeled | Valid JSON | Input (p50/p90) | Output (p50/p90) | Cost |")
    lines.append("|----------|---------|------------|-----------------|------------------|------|")

    total_cost = 0.0
    for provider, labels in labels_by_provider.items():
        labeled = len(labels)
        valid = sum(1 for o in labels.values() if o.chosen_synset_id)

        input_tokens = [o.usage.input_tokens for o in labels.values()]
        output_tokens = [o.usage.output_tokens for o in labels.values()]
        costs = [o.usage.cost_usd for o in labels.values()]

        inp_p50 = calculate_percentile(input_tokens, 50)
        inp_p90 = calculate_percentile(input_tokens, 90)
        out_p50 = calculate_percentile(output_tokens, 50)
        out_p90 = calculate_percentile(output_tokens, 90)
        cost = sum(costs)
        total_cost += cost

        lines.append(
            f"| {provider} | {labeled} | {valid} ({100*valid/labeled:.0f}%) | "
            f"{inp_p50:.0f}/{inp_p90:.0f} | {out_p50:.0f}/{out_p90:.0f} | ${cost:.4f} |"
        )

    lines.append(f"\n**Total pilot cost**: ${total_cost:.4f}")

    # Flag statistics
    lines.append("\n## Flag Statistics\n")
    lines.append("| Provider | needs_more_context | none_of_the_above | metaphor | multiword |")
    lines.append("|----------|--------------------|-------------------|----------|-----------|")

    for provider, labels in labels_by_provider.items():
        flag_counts = Counter()
        for output in labels.values():
            for flag in output.flags:
                flag_counts[flag] += 1

        nmc = flag_counts.get("needs_more_context", 0)
        nota = flag_counts.get("none_of_the_above", 0)
        meta = flag_counts.get("metaphor", 0)
        multi = flag_counts.get("multiword", 0)

        lines.append(
            f"| {provider} | {nmc} ({100*nmc/len(labels):.1f}%) | "
            f"{nota} ({100*nota/len(labels):.1f}%) | "
            f"{meta} ({100*meta/len(labels):.1f}%) | "
            f"{multi} ({100*multi/len(labels):.1f}%) |"
        )

    # Agreement analysis
    if len(labels_by_provider) >= 2:
        lines.append("\n## Inter-Model Agreement\n")

        # Get common example IDs
        provider_names = list(labels_by_provider.keys())
        common_ids = set(labels_by_provider[provider_names[0]].keys())
        for prov in provider_names[1:]:
            common_ids &= set(labels_by_provider[prov].keys())

        # Calculate agreement
        agree_count = 0
        disagree_examples = []

        for ex_id in common_ids:
            synsets = [labels_by_provider[p][ex_id].chosen_synset_id for p in provider_names]
            if len(set(synsets)) == 1:
                agree_count += 1
            else:
                disagree_examples.append((ex_id, synsets))

        agreement_rate = agree_count / len(common_ids) if common_ids else 0

        lines.append(f"- **Common examples**: {len(common_ids)}")
        lines.append(f"- **Full agreement**: {agree_count} ({100*agreement_rate:.1f}%)")
        lines.append(
            f"- **Disagreement**: {len(disagree_examples)} ({100*(1-agreement_rate):.1f}%)"
        )

        # Show some disagreements
        if disagree_examples:
            lines.append("\n### Sample Disagreements\n")
            lines.append("| Example ID | " + " | ".join(provider_names) + " |")
            lines.append("|------------|" + "|".join(["---"] * len(provider_names)) + "|")
            for ex_id, synsets in disagree_examples[:10]:
                short_id = ex_id.split("|")[-1]  # Just tok:N
                synset_cells = [s[:25] + "..." if len(s) > 25 else s for s in synsets]
                lines.append(f"| {short_id} | " + " | ".join(synset_cells) + " |")

    # Aggregation simulation
    if len(labels_by_provider) >= 2:
        lines.append("\n## Aggregation Simulation\n")

        gold_labels = []
        for ex_id in common_ids:
            outputs = [labels_by_provider[p][ex_id] for p in provider_names]
            label = aggregate_labels(outputs)
            gold_labels.append(label)

        stats = get_aggregation_stats(gold_labels)

        lines.append(f"- **Total labels**: {stats['total']}")
        lines.append(
            f"- **Needs referee**: {stats['needs_referee_count']} ({100*stats['needs_referee_ratio']:.1f}%)"
        )
        lines.append(f"- **Average agreement**: {stats['avg_agreement']:.2f}")
        lines.append(f"- **Average confidence**: {stats['avg_confidence']:.2f}")

    # Cost projections
    lines.append("\n## Cost Projections\n")
    lines.append("| Dataset Size | OpenAI | Anthropic | Total (2 providers) |")
    lines.append("|--------------|--------|-----------|---------------------|")

    for size in [1000, 2000, 3000]:
        scale = size / len(examples)
        openai_cost = 0.0
        anthropic_cost = 0.0

        if "openai" in labels_by_provider:
            openai_cost = (
                sum(o.usage.cost_usd for o in labels_by_provider["openai"].values()) * scale
            )
        if "anthropic" in labels_by_provider:
            anthropic_cost = (
                sum(o.usage.cost_usd for o in labels_by_provider["anthropic"].values()) * scale
            )

        total = openai_cost + anthropic_cost
        lines.append(f"| {size:,} | ${openai_cost:.2f} | ${anthropic_cost:.2f} | ${total:.2f} |")

    # Recommendations
    lines.append("\n## Recommendations\n")
    lines.append("Based on the pilot results:")
    lines.append("")

    if len(labels_by_provider) >= 2:
        if stats["needs_referee_ratio"] > 0.3:
            lines.append(
                "- ⚠️ High referee rate - consider adding 3rd model or adjusting thresholds"
            )
        else:
            lines.append("- ✅ Referee rate acceptable")

        if agreement_rate > 0.8:
            lines.append("- ✅ High inter-model agreement - good prompt quality")
        else:
            lines.append("- ⚠️ Low agreement - review disagreement examples")

    lines.append("")
    lines.append("---")
    lines.append("*End of pilot report*")

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"✅ Report saved to {output_path}")


@app.command()
def main(
    examples_path: Path = typer.Option(
        Path("data/wsd_gold/pilot_examples.jsonl"),
        "--examples",
        "-e",
        help="Path to examples JSONL",
    ),
    labels_dir: Path = typer.Option(
        Path("data/wsd_gold/labels"),
        "--labels",
        "-l",
        help="Directory with provider label files",
    ),
    output: Path = typer.Option(
        Path("reports/pilot_report.md"),
        "--output",
        "-o",
        help="Output report path",
    ),
) -> None:
    """Generate pilot report from labeling results."""
    # Load examples
    if not examples_path.exists():
        print(f"❌ Examples not found: {examples_path}")
        raise typer.Exit(1)

    examples = load_examples(examples_path)
    print(f"Loaded {len(examples)} examples")

    # Load labels from all providers
    labels_by_provider = {}
    for label_file in labels_dir.glob("*.jsonl"):
        provider = label_file.stem
        labels = load_labels(label_file)
        labels_by_provider[provider] = labels
        print(f"Loaded {len(labels)} labels from {provider}")

    if not labels_by_provider:
        print(f"❌ No label files found in {labels_dir}")
        raise typer.Exit(1)

    # Generate report
    generate_report(examples, labels_by_provider, output)


if __name__ == "__main__":
    app()
