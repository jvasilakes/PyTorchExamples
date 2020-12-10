Run the translation model with

```
python seq2seq.py <parameter_file>.json --verbose
```

See `example.json` for an example parameter file. A description of the expected fields is below:

```
"name": str, the name for this experiment
"data_path": str, path to the data file. See `data/eng-fra_small.txt` for an example.
"checkpoint_dir": str, where to save model checkpoints.
"reverse_input": bool, whether to reverse the input text as in the original Seq2Seq paper.
"embedding_dimension": int, the size of the embeddings
"num_layers": int, number of hidden layers in the encoder/decoder LSTM.
"epochs": int, number of epochs to train for.
"batch_size": int, the batch size. Currently ignored.
"grad_accumulation_steps": int, number of steps to accumulated the gradient.
"learn_rate": float, the optimizer learning rate.
"train": bool, whether to train the model.
"sample": bool, whether to output samples from the validation set.
"validate": bool, whether to evaluate the model on the validation set. Currently not implemented.
"test": bool, whether to evaluate the model on the test set. Currently not implemented.
```
