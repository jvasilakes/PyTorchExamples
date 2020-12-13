Generate the data with 

```
python get_data.py data/negation.jsonl data/sentences.jsonl
```

Run the language model with

```
python rnnlm.py <parameter_file>.jsonl --verbose
```

See `example.json` for an example parameter file. A description of the expected fields is below:

```
"name": str, the name for this experiment
"data_path": str, path to the data file. See `data/eng-fra_small.txt` for an example.
"glove_path": str, path to a file of GloVe embeddings.
"checkpoint_dir": str, where to save model checkpoints.
"epochs": int, number of epochs to train for.
"batch_size": int, the batch size. Currently ignored. Always 1.
"learn_rate": float, the optimizer learning rate.
"train": bool, whether to train the model.
"sample": bool, whether to output samples from the validation set.
"validate": bool, whether to evaluate the model on the validation set. Currently not implemented.
"test": bool, whether to evaluate the model on the test set. Currently not implemented.
```
