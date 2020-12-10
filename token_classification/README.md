Run the token classification model with

```
python token_classification.py <parameter_file>.json --verbose
```

See `example.json` for an example parameter file. A description of the expected fields is below:

```
"name": str, the name for this experiment
"data_dir": str, the path to the annotated data. See data/conll03 for an example.
"glove_path": str, path to the glove embeddings file.
"checkpoint_dir": str, directory in which to save model checkpoints.
"epochs": int, number of epochs to train for.
"batch_size": int, the batch size.
"learn_rate": float, the learning rate.
"train": bool, whether to run training.
"validate": bool, whether to evaluate on the validation set.
"test": bool, whether to evaluate on the test set.
```
