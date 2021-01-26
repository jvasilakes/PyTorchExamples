# Seq2Seq VAE

```
python vae.py config.json --verbose
```

Description of fields in config.json
```
name: str, the name of this experiment/model
random_seed: int, for reproducibility
data_dir: str, "/path/to/PyTorchExamples/seq2seq_vae/data/"
lowercase: bool, whether to lowercase the input
checkpoint_dir: str, "/path/to/PyTorchExamples/seq2seq_vae/model_checkpoints/",
embedding_dimension: int, the output dimension of the embedding layer. Not used if "glove_path" specified.
glove_path: str, "/path/to/glove.6B.*d.txt"
hidden_dimension: int, the hidden dimsension of the LSTM layers in the encoder and decoder
num_layers: int, the number of LSTM layers in the encoder and decoder
latent_dimension: int, the dimensionality of the latent space z
epochs: int, number of epochs to train for
batch_size: int, number of examples to batch
learn_rate: float, learning rate for the ADAM optimizer
dropout: float, governs dropout between LSTM layers and the output of the embedding dimensions.
objective": str, "beta-vae" (L_reconstruct + beta * KL_div) or "constraint-vae" (L_reconstruct + beta * abs(KL_div - C))
teacher_forcing_prob": float, 0-1 probability of using the ground truth token for each decoding step
max_nats: float, corresponds to C in constraint-vae objective
nats_increase_steps": int, number of training steps over which to increase C up to max_nats
beta: float, beta hyperparameter in objectives
train: bool, whether to run training
validate: bool, whether to run evaluation on dev set
test: bool, whether to run evaluation on test set
sample: bool, whether to generate samples from the posterior
```

Plot the posterior z values:
```
python plot_zs.py logs/example/zs/ train logs/example/zs/plots
```
