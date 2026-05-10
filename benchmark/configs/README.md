# Benchmark Configs Directory

This directory stores Ludwig config files for each benchmark dataset.

## Structure

```
benchmark/configs/
├── {dataset_name}/
│   ├── configs.jsonl       # One Ludwig config dict per line (JSON Lines format)
│   ├── schema.json         # Feature schema: input_features + output_feature
│   └── sampler_meta.json   # Seed, n_generated, n_valid, timestamp
└── README.md               # This file
```

## Format: configs.jsonl

Each line is a complete Ludwig config dict:

```json
{"input_features": [...], "output_features": [...], "combiner": {...}, "trainer": {...}}
```

## Generating configs

```bash
python scripts/generate_configs.py --registry benchmark/dataset_registry.json --configs-dir benchmark/configs --n 100
```

## Config diversity guarantee

The sampler ensures at least 1 config per valid combiner type, then distributes
remaining budget proportionally. Configs are deduplicated by SHA-256 of their
canonical JSON representation.

## Valid combiner types

tabnet (tabular only), transformer, tabtransformer, ft_transformer, concat,
project_aggregate, comparator (2-input only), sequence/sequence_concat (sequential inputs only),
cross_attention, perceiver, gated_fusion, hypernetwork, tabpfn_v2 (tabular only)
