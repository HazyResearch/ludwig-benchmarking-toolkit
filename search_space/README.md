# Search Space

This directory defines the AutoML search space for Ludwig benchmarking as plain YAML files. The loader scans these directories at runtime — adding or removing a YAML file is all that is needed to extend or prune the space.

## Directory layout

```
search_space/
  encoders/       one file per encoder
  combiners/      one file per combiner
  decoders/       one file per decoder
  trainer.yaml    global trainer hyperparameters
```

---

## Encoder schema

```yaml
name: dense                                    # Ludwig encoder type string
feature_types: [binary, number, category]      # input feature types this encoder is valid for
preprocessing:                                 # optional: fixed preprocessing overrides applied when this
  word_tokenizer: hf_tokenizer                 #   encoder is selected (used for HuggingFace encoders)
  pretrained_model_name_or_path: bert-base-uncased
hyperparameters:                               # parameters to sample; omit or leave empty if none
  num_layers:
    values: [1, 2, 3]                          # categorical grid — one value is sampled per trial
  dropout:
    values: [0.0, 0.1, 0.3]
```

`feature_types` controls which input features this encoder is a candidate for. An encoder is only considered when the feature's type appears in this list.

`preprocessing` keys are merged verbatim into the feature's preprocessing config when the encoder is selected.

`hyperparameters` is a map from parameter name to a sampling spec. Currently the only supported spec is `values`, which defines a discrete set to sample from. A parameter with a single-element `values` list is effectively a fixed override.

To add a new encoder, drop a new YAML file in `encoders/`. The filename is used only for human reference; the `name` field is what gets written into the Ludwig config.

---

## Combiner schema

```yaml
name: tabnet
constraints:                                   # optional: all constraints must hold for this combiner
  requires_all_tabular: true                   #   all input features must be tabular types
  requires_sequential: true                    #   at least one input feature must be sequential/text
  exact_n_inputs: 2                            #   the model must have exactly this many input features
hyperparameters:
  size:
    values: [8, 16, 32]
```

`constraints` is optional. Omit it entirely (or omit individual keys) when there are no restrictions.

- `requires_all_tabular`: combiners like `tabnet`, `tabtransformer`, `ft_transformer`, and `tabpfn_v2` only make sense when every input feature is a tabular type (binary, number, category). Tabular types are defined by the loader.
- `requires_sequential`: combiners like `sequence` and `sequence_concat` require at least one sequential or text input feature.
- `exact_n_inputs`: `comparator` requires exactly two input features.

To add a new combiner, drop a new YAML file in `combiners/`. If it has no constraints, omit the `constraints` key.

---

## Decoder schema

```yaml
name: mlp_classifier
feature_types: [binary, category]             # output feature types this decoder handles
hyperparameters:
  num_layers:
    values: [1, 2]
  dropout:
    values: [0.0, 0.1]
```

`feature_types` restricts which output features this decoder is a candidate for. The loader matches decoders to each output feature by type.

To add a new decoder, drop a new YAML file in `decoders/`.

---

## Trainer schema (`trainer.yaml`)

```yaml
learning_rate:
  values: [1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2]
batch_size:
  values: [64, 128, 256, 512]
epochs:
  default: 50                                  # fixed value, not sampled
```

Each key is a trainer parameter. Use `values` for a sampled grid or `default` for a fixed value. There is a single `trainer.yaml`; it is not split by model type.

---

## Sampling spec reference

| Key | Meaning |
|-----|---------|
| `values` | Sample one element uniformly from this list per trial |
| `default` | Use this fixed value; do not include in the search grid |

---

## Adding a new component

1. Create a YAML file in the appropriate subdirectory.
2. Set `name` to the Ludwig type string the loader will inject into the config.
3. Set `feature_types` (encoders/decoders) or `constraints` (combiners) as needed.
4. Populate `hyperparameters` or leave it as `{}`.
5. No code changes required — the loader discovers files by glob.
