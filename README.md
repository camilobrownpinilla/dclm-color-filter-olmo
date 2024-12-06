
# DCLM-CoLoR

## Preparing data
To tokenize raw text, see [dclm docs](https://github.com/mlfoundations/dclm?tab=readme-ov-file#tokenize-and-shuffle)


## Configuring pipeline
*Requirements:* .tar archive(s) of tokenized text (see above)

Set configs in `./dclm_color_filter_olmo/configs/` to configure pipeline.


```
dclm/format.yaml:
```

- `tar_paths`: path to tokenized tokens

- `memmap_path`: where to store memmaps (used by CF)


```
sweeps/color-1b.yaml & sweeps/finetune-dclm.yaml:
```

- Set model parameters. Can leave alone.

```
sweeps/score-dclm.yaml:
```

- `checkpoints_path`: Where to save model outs.

```
sweeps/combine_sort.yaml:
```
- `checkpoints_path`, `prior_path`, `conditional_path`: path to directory with model data and names of models

```
dclm/select_topk.yaml:
```
- Configure path to sorted scores, destination for selected data, and number and mode of selection (documents or chunks). 

## Running pipeline
Run entire pipeline to train models and select data for DCLM. 

From root directory,
```
./dclm_pipeline.sh
```
to skip training
```
./dclm_pipeline.sh [--skip-training]
```

## Training and evaluating DCLM
From root directory, run
```
sbatch run_dclm.sh
```
Configure flags and SLURM info as necessary.



