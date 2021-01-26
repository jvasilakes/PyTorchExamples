import os
import re
import csv
import json
import pickle
import random
import logging
import argparse
import torch
import torch.nn as nn
import numpy as np

from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchtext.data.metrics import bleu_score

import texar.torch as tx

"""
An LSTM Seq2Seq model with variational inference for text generation.
"""

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("params_json", type=str,
                        help="""Path to JSON file containing experiment
                                parameters.""")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="""If specified, print tqdm progress bars
                                for training and evaluation.""")
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class VariationalEncoder(nn.Module):
    """
    embedding -> dropout -> LSTM
    """
    def __init__(self, vocab_size, emb_dim, hidden_size, num_layers,
                 dropout_rate=0.5, emb_matrix=None):
        super(VariationalEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        if emb_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(
                    torch.tensor(emb_matrix))
            self.embedding.weight.requires_grad = False
            self.vocab_size, self.emb_dim = emb_matrix.shape
        else:
            self.embedding = nn.Embedding(self.vocab_size, self.emb_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.recurrent = nn.LSTM(self.emb_dim, self.hidden_size,
                                 num_layers=self.num_layers,
                                 dropout=self.dropout_rate, batch_first=True)

    def forward(self, inputs, lengths, hidden):
        # inputs: [batch_size, max(lengths)]
        embedded = self.dropout(self.embedding(inputs))
        # embedded: [batch_size, max(lengths), self.emb_dim]
        packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False)
        # packed: [sum(lengths), self.emb_dim]
        encoded, hidden = self.recurrent(packed, hidden)
        # encoded: [batch_size, max(lengths), self.hidden_size]
        # hidden: [num_layers, batch_size, hidden_size]
        unpacked, lengths_unpacked = nn.utils.rnn.pad_packed_sequence(
                encoded, batch_first=True)
        # unpacked: [batch_size, max(lengths), hidden_size]
        return unpacked, hidden

    def init_hidden(self, batch_size, device=DEVICE):
        # Initialize the LSTM state.
        # One for hidden and one for the cell
        return (torch.zeros(self.num_layers, batch_size,
                            self.hidden_size, device=device),
                torch.zeros(self.num_layers, batch_size,
                            self.hidden_size, device=device))


class VariationalDecoder(nn.Module):
    """
    LSTM -> linear -> token_predictions
    """
    def __init__(self, vocab_size, emb_dim, hidden_size,
                 num_layers, dropout_rate=0.5, emb_matrix=None):
        super(VariationalDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        if emb_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(
                    torch.tensor(emb_matrix))
            self.embedding.weight.requires_grad = False
            self.vocab_size, self.emb_dim = emb_matrix.shape
        else:
            self.embedding = nn.Embedding(self.vocab_size, self.emb_dim)

        self.dropout = nn.Dropout(self.dropout_rate)
        self.recurrent = nn.LSTM(self.emb_dim,
                                 self.hidden_size,
                                 num_layers=self.num_layers,
                                 dropout=self.dropout_rate, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, inputs, lengths, hidden):
        embedded = self.dropout(self.embedding(inputs))
        # embedded: [batch_size, len(inputs), emb_dim]
        packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False)
        # packed: [sum(lengths), self.emb_dim]
        decoded, hidden = self.recurrent(packed, hidden)
        # decoded: [batch_size, max(lengths), self.hidden_size]
        # hidden: [num_layers, batch_size, hidden_size]
        unpacked, lengths_unpacked = nn.utils.rnn.pad_packed_sequence(
                decoded, batch_first=True)
        # logits: [batch_size, len(inputs), vocab_size]
        logits = self.linear(unpacked)
        return logits, hidden


class VariationalSeq2Seq(nn.Module):
    """
    LSTM_encoder -> final_state -> LSTM_decoder
    """
    def __init__(self, encoder, decoder, latent_dim,
                 sos_token_idx, eos_token_idx):
        super(VariationalSeq2Seq, self,).__init__()
        assert isinstance(encoder, VariationalEncoder)
        assert isinstance(decoder, VariationalDecoder)
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.context2params = nn.Linear(
                encoder.hidden_size * encoder.num_layers, 2 * self.latent_dim)
        self.z2hidden = nn.Linear(
                self.latent_dim, 2 * decoder.hidden_size * decoder.num_layers)
        self.sos_token_idx = sos_token_idx
        self.eos_token_idx = eos_token_idx

    def trainable_parameters(self):
        return [param for param in self.parameters()
                if param.requires_grad is True]

    def encode(self, inputs, lengths):
        # encoder_hidden: tuple([num_layers, batch_size, hidden_size])
        batch_size = inputs.size(0)
        encoder_hidden = self.encoder.init_hidden(batch_size)
        encoded, encoder_hidden = self.encoder(inputs, lengths, encoder_hidden)
        # context: [batch_size, num_layers * hidden_size]
        context = encoder_hidden[0].view(batch_size, -1)
        return encoded, context

    def compute_z(self, context):
        # params: [batch_size, 2 * latent_dim]
        params = self.context2params(context)
        # mu/log_sigma: [batch_size, latent_dim]
        mu, log_sigma = params.chunk(2, dim=1)
        # Sample from N(mu, log_sigma) using the reparameterization trick
        # z: [batch_size, latent_dim]
        z = mu + torch.randn_like(log_sigma) * torch.exp(log_sigma)
        return z, mu, log_sigma

    def compute_hidden(self, z, batch_size):
        # hidden: [batch_size, 2 * hidden_size * decoder.num_layers]
        hidden = torch.tanh(self.z2hidden(z))
        # state/cell: [batch_size, hidden_size * decoder.num_layers]
        state, cell = hidden.chunk(2, dim=1)
        # state/cell: [num_layers, batch_size, hidden_size]
        state = state.reshape(self.decoder.num_layers, batch_size, -1)
        cell = cell.reshape(self.decoder.num_layers, batch_size, -1)
        decoder_hidden = (state, cell)
        return decoder_hidden

    def forward(self, inputs, lengths, teacher_forcing_prob=0.5):
        # inputs: [batch_size, max(lengths)]
        encoded, context = self.encode(inputs, lengths)

        z, mu, log_sigma = self.compute_z(context)
        batch_size = inputs.size(0)
        decoder_hidden = self.compute_hidden(z, batch_size)

        # decoder_input: [batch_size, 1]
        decoder_input = torch.LongTensor([[self.sos_token_idx]]).to(DEVICE)
        decoder_input = decoder_input.repeat(batch_size, 1)
        input_lengths = [1] * batch_size
        vocab_size = self.decoder.vocab_size
        target_length = inputs.size(-1)
        # Placeholder for predictions
        decoder_output = torch.zeros(
                batch_size, target_length, vocab_size).to(DEVICE)
        decoder_output[:, 0, self.sos_token_idx] = 1.0
        for i in range(1, target_length):
            # logits: [batch_size, 1, vocab_size]
            logits, decoder_hidden = self.decoder(
                    decoder_input, input_lengths, decoder_hidden)
            decoder_output[:, i, :] = logits.squeeze()
            use_teacher_forcing = random.random() < teacher_forcing_prob
            if use_teacher_forcing is True:
                # We lose a dimension slicing outputs, so we add it back.
                target = inputs[:, i]
                decoder_input = torch.unsqueeze(target, 1)
            else:
                # Argmax over the vocab.
                probs = torch.softmax(logits, dim=-1)
                decoder_input = probs.argmax(-1).detach()
        return decoder_output, mu, log_sigma, z

    def sample(self, z, max_length=30):
        hidden = torch.tanh(self.z2hidden(z))
        state, cell = hidden.chunk(2, dim=1)
        state = state.reshape(self.decoder.num_layers, 1, -1)
        cell = cell.reshape(self.decoder.num_layers, 1, -1)
        decoder_hidden = (state, cell)

        decoder_input = torch.LongTensor([[self.sos_token_idx]]).to(DEVICE)
        input_lengths = [1]
        # Placeholder for predictions
        decoder_output = [self.sos_token_idx]
        for i in range(max_length):
            # logits: [batch_size, 1, vocab_size]
            logits, decoder_hidden = self.decoder(
                    decoder_input, input_lengths, decoder_hidden)
            probs = torch.softmax(logits, dim=-1)
            decoder_input = probs.argmax(-1).detach()
            decoder_output.append(decoder_input.item())
            if decoder_input.item() == self.eos_token_idx:
                break
        return decoder_output


def factorized_kl_divergence(mu, log_sigma):
    kl_losses = 0.5 * (torch.exp(log_sigma) +
                       torch.pow(mu, 2) - 1 - log_sigma)
    return kl_losses


def vae_loss(inputs, reconstruction, lengths, mu, log_sigma):
    kl_loss = 0.5 * (torch.exp(log_sigma) +
                     torch.pow(mu, 2) - 1 - log_sigma)
    # Average over the batch, then sum over the dimensions
    kl_loss = kl_loss.mean(0).sum()
    recon_loss = tx.losses.sequence_sparse_softmax_cross_entropy(
            labels=inputs, logits=reconstruction, sequence_length=lengths)
    return kl_loss, recon_loss


def load_glove(path):
    """
    Load the GLoVe embeddings from the provided path.
    Return the embedding matrix and the embedding dimension.
    Pickles the loaded embedding matrix for fast loading
    in the future.

    :param str path: Path to the embeddings. E.g.
                     `glove.6B/glove.6B.100d.txt`
    :returns: embeddings, embedding_dim
    :rtype: Tuple(numpy.ndarray, int)
    """
    bn = os.path.splitext(os.path.basename(path))[0]
    pickle_file = bn + ".pickle"
    if os.path.exists(pickle_file):
        logging.warning(f"Loading embeddings from pickle file {pickle_file} in current directory.")  # noqa
        glove = pickle.load(open(pickle_file, "rb"))
        emb_dim = list(glove.values())[0].shape[0]
        return glove, emb_dim

    vectors = []
    words = []
    idx = 0
    word2idx = {}

    with open(path, "rb") as inF:
        for line in inF:
            line = line.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)
    emb_dim = vect.shape[0]
    glove = {word: np.array(vectors[word2idx[word]]) for word in words}
    if not os.path.exists(pickle_file):
        pickle.dump(glove, open(pickle_file, "wb"))
    return glove, emb_dim


def get_embedding_matrix(vocab, glove):
    emb_dim = len(list(glove.values())[0])
    matrix = np.zeros((len(vocab), emb_dim), dtype=np.float32)
    found = 0
    for (i, word) in enumerate(vocab):
        try:
            matrix[i] = glove[word]
            found += 1
        except KeyError:
            matrix[i] = np.random.normal(scale=0.6, size=(emb_dim,))
    logging.info(f"Found {found}/{len(vocab)} vocab words in embedding.")
    word2idx = {word: idx for (idx, word) in enumerate(vocab)}
    return matrix, word2idx


class TextDataset(torch.utils.data.Dataset):
    """
    A basic dataset for language modeling or autoencoding.
    """
    def __init__(self, docs, word2idx):
        self.docs = docs
        self.word2idx = word2idx
        self.Xs = [self.doc2tensor(doc) for doc in self.docs]
        if "<UNK>" not in word2idx.keys():
            raise ValueError("word2idx must have an '<UNK>' entry.")
        if "<PAD>" not in word2idx.keys():
            raise ValueError("word2idx must have an '<PAD>' entry.")

    def __getitem__(self, idx):
        return self.Xs[idx]

    def __len__(self):
        return len(self.Xs)

    def doc2tensor(self, doc):
        idxs = []
        for w in doc:
            try:
                idxs.append(self.word2idx[w])
            except KeyError:
                idxs.append(self.word2idx["<UNK>"])
        return torch.LongTensor([[idxs]])


def get_sentences(path):
    """
    Read the output of get_data.py
    """
    sentences = []
    with open(path, 'r') as inF:
        for line in inF:
            data = json.loads(line)
            sentences.append(data["sentence"])
    return sentences


def pad_sequence(batch):
    """
    Pad the sequence batch with 0 vectors for X.
    Meant to be used as the value of the `collate_fn`
    argument to `torch.utils.data.DataLoader`.
    """
    seqs = [torch.squeeze(x) for x in batch]
    lengths = [len(s) for s in seqs]
    seqs_padded = nn.utils.rnn.pad_sequence(
            seqs, batch_first=True, padding_value=0)
    return seqs_padded, lengths


def preprocess_sentences(sentences, SOS, EOS, lowercase=True):
    """
    Preprocess the input text by
      (optional) lowercasing
      adding space around punctuation
      removing strange characters
      tokenizing on whitespace
      (optional) reversing the words as suggested by the Seq2Seq paper
      adding <SOS> and <EOS> tokens
    """
    out_data = []
    for sent in sentences:
        sent = sent.strip()
        if lowercase is True:
            sent = sent.lower()
        sent = re.sub(r"([.!?])", r" \1", sent)
        sent = re.sub(r"[^a-zA-Z.!?]+", r" ", sent)
        sent = sent.split()  # Tokenize
        sent = [SOS] + sent + [EOS]
        out_data.append(sent)
    return out_data


def load_latest_checkpoint(model, optimizer, checkpoint_dir):
    """
    Find the most recent (in epochs) checkpoint in checkpoint dir and load
    it into the model and optimizer. Return the model and optimizer
    along with the epoch the checkpoint was trained to.
    If not checkpoint is found, return the unchanged model and optimizer,
    and 0 for the epoch.
    """
    ls = os.listdir(checkpoint_dir)
    ckpts = [fname for fname in ls if fname.endswith(".pt")]
    if ckpts == []:
        return model, optimizer, 0, None

    latest_ckpt_idx = 0
    latest_epoch = 0
    for (i, ckpt) in enumerate(ckpts):
        epoch = ckpt.replace("model_", '').replace(".pt", '')
        epoch = int(epoch)
        if epoch > latest_epoch:
            latest_epoch = epoch
            latest_ckpt_idx = i

    ckpt = torch.load(os.path.join(checkpoint_dir, ckpts[latest_ckpt_idx]))
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    next_epoch = ckpt["epoch"] + 1
    return model, optimizer, next_epoch, ckpts[latest_ckpt_idx]


def get_kl_weight(step, total_steps, warmup_steps=0):
    """
    For KL annealing
    """
    weight = 0.0
    if step > warmup_steps:
        step = step - warmup_steps
        weight = min(0.5, step / total_steps)
    return weight


def get_kl_limit(step, total_steps, max_nats):
    """
    For constraining the KL divergence to a specific value
    as in http://arxiv.org/abs/1804.03599
    """
    limit = min(1.0, step / total_steps) * max_nats
    return limit


def tensor2text(tensor, idx2word, eos_token_idx):
    try:
        eos = torch.where(tensor == eos_token_idx)[0][0]
    except IndexError:
        eos = tensor.size(0)
    return [idx2word[i.item()] for i in tensor[:eos+1]]


def get_reconstructions(model, dataset, idx2word, idxs):
    batch = [dataset[i] for i in idxs]
    Xs, lengths = pad_sequence(batch)
    Xs = Xs.to(DEVICE)
    lengths = torch.tensor(lengths).to(DEVICE)
    logits, mu, log_sigma, z = model(Xs, lengths)

    X_text = [' '.join(tensor2text(X, idx2word, model.eos_token_idx))
              for X in Xs.cpu().detach()]
    recon_idxs = logits.argmax(-1)
    recon_text = [' '.join(tensor2text(r, idx2word, model.eos_token_idx))
                  for r in recon_idxs]
    joined = '\n'.join([f"'{x}' ==> '{r}'" for (x, r)
                        in zip(X_text, recon_text)])
    return joined


def log_reconstructions(model, dataset, idx2word, name, epoch, logdir, n=10):
    idxs = np.random.choice(len(dataset),
                            size=n,
                            replace=False)
    # Log inputs and their reconstructions before model training
    recon_file = os.path.join(logdir, f"reconstructions_{name}.log")
    recon_str = get_reconstructions(model, dataset, idx2word, idxs)
    with open(recon_file, 'a') as outF:
        outF.write(f"EPOCH {epoch}\n")
        outF.write(recon_str + '\n')


def measure_autoencoding(model, dataloader, epoch, summary_writer):
    mses = []
    for (i, (Xbatch, lengths)) in enumerate(dataloader):
        Xbatch = Xbatch.to(DEVICE)
        lengths = torch.tensor(lengths).to(DEVICE)
        logits, mu, log_sigma, z = model(
                Xbatch, lengths, teacher_forcing_prob=0.0)
        x_prime = logits.argmax(-1)
        logits_p, mu_p, log_sigma_p, z_p = model(
                x_prime, lengths, teacher_forcing_prob=0.0)
        z_mse = torch.nn.functional.mse_loss(z, z_p)
        mses.append(z_mse.item())
    summary_writer.add_scalar("AE_MSE", np.mean(mses), epoch)
    summary_writer.flush()


def compute_bleu(Xbatch, pred_batch, idx2word, eos_token_idx):
    Xtext = [[tensor2text(X, idx2word, eos_token_idx)[1:-1]]  # Rm SOS and EOS
             for X in Xbatch.cpu().detach()]
    pred_text = [tensor2text(pred, idx2word, eos_token_idx)[1:-1]
                 for pred in pred_batch.cpu().detach()]
    bleu = bleu_score(pred_text, Xtext)
    return bleu


def train(model, opt, dataloader, loss_fn, epoch, params,
          idx2word, verbose=True, summary_writer=None):
    """
    Perform a single training epoch on dataloader.
    Save the metrics to a Tensorboard logdir specified by summary_writer.
    """

    if summary_writer is None:
        summary_writer = SummaryWriter()

    model.train()
    kl_losses = []
    recon_losses = []
    bleus = []
    zs = []
    zs_meta = []
    if verbose is True:
        pbar = tqdm(total=len(dataloader))
    step = epoch * len(dataloader)
    for (i, (Xbatch, lengths)) in enumerate(dataloader):
        Xbatch = Xbatch.to(DEVICE)
        lengths = torch.tensor(lengths).to(DEVICE)
        logits, mu, log_sigma, z = model(
                Xbatch, lengths,
                teacher_forcing_prob=params["teacher_forcing_prob"])
        kl_loss, recon_loss = loss_fn(Xbatch, logits, lengths, mu, log_sigma)
        C = get_kl_limit(step, params["nats_increase_steps"],
                         params["max_nats"])
        # Objective taken from
        # Understanding Disentangling in beta-VAE (Burgess et al 2018), eq 8.
        beta = params["beta"]
        if params["objective"].lower() == "beta-vae":
            loss = recon_loss + beta * kl_loss
        elif params["objective"].lower() == "constraint-vae":
            loss = recon_loss + beta * torch.abs(kl_loss - C)
        loss.backward()
        kl_losses.append(kl_loss.item())
        recon_losses.append(recon_loss.item())
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1)
        opt.step()
        opt.zero_grad()

        bleus.append(compute_bleu(Xbatch, logits.argmax(-1), idx2word,
                                  model.eos_token_idx))
        zs.append(z.cpu().detach().numpy())
        zs_meta.extend([' '.join(tensor2text(X, idx2word, model.eos_token_idx))
                        for X in Xbatch.cpu().detach()])

        if verbose is True:
            pbar.update(1)
            pbar.set_description(f"({epoch}) KL: {kl_loss:.4f}|{C:.2f} CE: {recon_loss:.4f}")  # noqa
        if step % 10 == 0:
            summary_writer.add_scalar("C", C, step)
            summary_writer.add_scalar("kl_step", kl_loss.item(), step)
            summary_writer.add_scalar("ce_step", recon_loss.item(), step)
        step += 1
    if verbose is True:
        pbar.close()

    zs = np.concatenate(zs)
    summary_writer.add_embedding(zs, metadata=zs_meta,
                                 global_step=step, tag="zs")
    summary_writer.add_scalar("avg_kl", np.mean(kl_losses), epoch)
    summary_writer.add_scalar("avg_ce", np.mean(recon_losses), epoch)
    summary_writer.add_scalar("avg_bleu", np.mean(bleus), epoch)
    summary_writer.flush()
    logstr = f"TRAIN ({epoch}) KL: {np.mean(kl_losses):.4f} +/- {np.std(kl_losses):.4f}"  # noqa
    logstr += f" | CE: {np.mean(recon_losses):.4f} +/- {np.std(recon_losses):.4f}"  # noqa
    logging.info(logstr)
    return model, opt


def evaluate(model, dataset, loss_fn, epoch, idx2word, name="dev",
             verbose=True, summary_writer=None):
    """
    Perform a single evaluation loop on dataset.
    Save the metrics to a Tensorboard logdir specified by summary_writer.
    """
    model.eval()
    kl_losses = []
    recon_losses = []
    bleus = []
    if verbose is True:
        pbar = tqdm(total=len(dataset))
    for (i, (Xbatch, lengths)) in enumerate(dataset):
        Xbatch = Xbatch.to(DEVICE)
        lengths = torch.tensor(lengths).to(DEVICE)
        logits, mu, log_sigma, z = model(Xbatch, lengths,
                                         teacher_forcing_prob=0.0)
        # Flatten the target and the reconstruction for the loss function
        kl_loss, recon_loss = loss_fn(Xbatch, logits, lengths, mu, log_sigma)
        kl_losses.append(kl_loss.item())
        recon_losses.append(recon_loss.item())
        bleus.append(compute_bleu(Xbatch, logits.argmax(-1), idx2word,
                                  model.eos_token_idx))

        if verbose is True:
            pbar.update(1)
            pbar.set_description(f"(EVAL: {epoch}) KL: {kl_loss:.4f} CE: {recon_loss:.4f}")  # noqa
    if verbose is True:
        pbar.close()

    if summary_writer is not None:
        summary_writer.add_scalar("avg_kl", np.mean(kl_losses), epoch)
        summary_writer.add_scalar("avg_ce", np.mean(recon_losses), epoch)
        summary_writer.add_scalar("avg_bleu", np.mean(bleus), epoch)
        summary_writer.flush()
    logstr = f"{name.upper()} ({epoch}) KL: {np.mean(kl_losses):.4f} +/- {np.std(kl_losses):.4f}"  # noqa
    logstr += f" | CE: {np.mean(recon_losses):.4f} +/- {np.std(recon_losses):.4f}"  # noqa
    logging.info(logstr)


def compute_z_metadata(model, dataset):
    zs = None
    mus = None
    kls = None
    for (i, (Xbatch, lengths)) in enumerate(dataset):
        Xbatch = Xbatch.to(DEVICE)
        lengths = torch.tensor(lengths).to(DEVICE)
        _, mu, log_sigma, z = model(Xbatch, lengths)
        kl = factorized_kl_divergence(mu, log_sigma)
        if i == 0:
            zs = z.detach().cpu().numpy()
            mus = mu.detach().cpu().numpy()
            kls = kl.detach().cpu().numpy()
        else:
            zs = np.concatenate([zs, z.detach().cpu().numpy()])
            mus = np.concatenate([mus, mu.detach().cpu().numpy()])
            kls = np.concatenate([kls, kl.detach().cpu().numpy()])
    aus = np.std(zs, axis=0)
    kls_mean = np.mean(kls, axis=0)
    return zs, aus, kls_mean


def log_z_metadata(model, dataloader, name, epoch,
                   logdir, summary_writer):
    os.makedirs(os.path.join(logdir, "zs"), exist_ok=True)
    zs_file = os.path.join(logdir, "zs", f"zs_{name}_{epoch}.log")
    aus_file = os.path.join(logdir, f"active_units_{name}.log")
    kls_file = os.path.join(logdir, f"KLs_{name}.log")

    zs, aus, kls = compute_z_metadata(model, dataloader)

    # Log z values per epoch so we can plot the approximate posterior
    with open(zs_file, 'w') as outF:
        writer = csv.writer(outF, delimiter=',')
        for z_row in zs:
            z_row = [f"{z:.4f}" for z in z_row]
            writer.writerow(z_row)

    summary_writer.add_histogram(
            f"var z: {name}", aus, global_step=epoch)

    header = None
    if not os.path.exists(aus_file):
        header = ["EPOCH"] + [f"var_{i}" for i in range(len(aus))]
    au_row = [epoch] + [f"{au:.4f}" for au in aus]
    kl_row = [epoch] + [f"{kl:.4f}" for kl in kls]
    with open(aus_file, 'a') as outF:
        writer = csv.writer(outF, delimiter=',')
        if header is not None:
            writer.writerow(header)
        writer.writerow(au_row)

    with open(kls_file, 'a') as outF:
        writer = csv.writer(outF, delimiter=',')
        if header is not None:
            writer.writerow(header)
        writer.writerow(kl_row)


def validate_params(params):
    valid_params = {
            "name": str,
            "random_seed": int,
            "data_dir": str,
            "lowercase": bool,
            "checkpoint_dir": str,
            "embedding_dimension": int,
            "glove_path": str,
            "hidden_dimension": int,
            "num_layers": int,
            "latent_dimension": int,
            "epochs": int,
            "batch_size": int,
            "learn_rate": float,
            "dropout": float,
            "objective": str,
            "teacher_forcing_prob": float,
            "max_nats": float,
            "nats_increase_steps": int,
            "beta": float,
            "train": bool,
            "sample": bool,
            "validate": bool,
            "test": bool}
    valid = True
    for (key, val) in valid_params.items():
        if key not in params.keys():
            logging.critical(f"parameter file missing '{key}'")
            valid = False
        if not isinstance(params[key], val):
            param_type = type(params[key])
            logging.critical(f"Parameter '{key}' of incorrect type!")
            logging.critical(f"  Expected '{val}' but got '{param_type}'.")
            valid = False
    if params["objective"].lower() not in ["beta-vae", "constraint-vae"]:
        valid = False
        logging.critical(f"Unknown objective function '{params['objective']}'")
    if valid is False:
        raise ValueError("Found incorrectly specified parameters.")

    for key in params.keys():
        if key not in valid_params.keys():
            logging.warning(f"Ignoring unused parameter '{key}' in parameter file.")  # noqa


def run(params_file, verbose=False):
    SOS = "<SOS>"
    EOS = "<EOS>"

    params = json.load(open(params_file, 'r'))
    validate_params(params)
    set_seed(params["random_seed"])

    logdir = os.path.join("logs", params["name"])
    os.makedirs(logdir, exist_ok=True)
    logfile = os.path.join(logdir, "run.log")
    print(f"Logging to {logfile}")
    logging.basicConfig(filename=logfile, level=logging.INFO)

    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H:%M:%S")
    logging.info(f"START: {now_str}")
    logging.info("PARAMETERS:")
    for (param, val) in params.items():
        logging.info(f"  {param}: {val}")

    ckpt_dir = os.path.join(params["checkpoint_dir"], params["name"])
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    # Read train/dev/test data
    train_file = os.path.join(params["data_dir"], "train.jsonl")
    dev_file = os.path.join(params["data_dir"], "dev.jsonl")
    test_file = os.path.join(params["data_dir"], "test.jsonl")

    train_sents = get_sentences(train_file)
    do_lowercase = params["lowercase"]
    train_sents = preprocess_sentences(train_sents, SOS, EOS,
                                       lowercase=do_lowercase)

    dev_sents = get_sentences(dev_file)
    dev_sents = preprocess_sentences(dev_sents, SOS, EOS,
                                     lowercase=do_lowercase)

    test_sents = get_sentences(test_file)
    test_sents = preprocess_sentences(test_sents, SOS, EOS,
                                      lowercase=do_lowercase)

    # Get the vocabulary from the training data
    vocab = ["<PAD>", "<UNK>"] + \
        list(sorted({word for doc in train_sents for word in doc}))
    word2idx = {word: idx for (idx, word) in enumerate(vocab)}
    # Load the glove embeddings, if specified
    emb_matrix = None
    if params["glove_path"] != "":
        logging.info(f"Loading embeddings from {params['glove_path']}")
        glove, _ = load_glove(params["glove_path"])
        emb_matrix, word2idx = get_embedding_matrix(vocab, glove)
        logging.info(f"Loaded embeddings with size {emb_matrix.shape}")
    idx2word = {idx: word for (word, idx) in word2idx.items()}

    # Get the train/dev/test dataloaders.
    if params["train"] is True or params["sample"] is True:
        # Load the training dataset.
        train_data = TextDataset(train_sents, word2idx)
        train_dataloader = torch.utils.data.DataLoader(
                train_data, shuffle=True, batch_size=params["batch_size"],
                collate_fn=pad_sequence)
        logging.info(f"Training examples: {len(train_sents)}")
        train_writer_path = os.path.join("runs", params["name"], "train")
        train_writer = SummaryWriter(log_dir=train_writer_path)

    if params["validate"] is True or params["sample"] is True:
        dev_data = TextDataset(dev_sents, word2idx)
        dev_dataloader = torch.utils.data.DataLoader(
                dev_data, shuffle=False,
                batch_size=params["batch_size"],
                collate_fn=pad_sequence)
        logging.info(f"Validation examples: {len(dev_sents)}")
        dev_writer_path = os.path.join("runs", params["name"], "dev")
        dev_writer = SummaryWriter(log_dir=dev_writer_path)

    if params["test"] is True:
        test_data = TextDataset(test_sents, word2idx)
        test_dataloader = torch.utils.data.DataLoader(
                test_data, shuffle=False,
                batch_size=params["batch_size"],
                collate_fn=pad_sequence)
        logging.info(f"Testing examples: {len(test_sents)}")

    # Build the model
    emb_dim = params["embedding_dimension"]
    hidden_size = params["hidden_dimension"]
    encoder = VariationalEncoder(len(vocab), emb_dim, hidden_size,
                                 params["num_layers"],
                                 dropout_rate=params["dropout"],
                                 emb_matrix=emb_matrix)
    encoder.to(DEVICE)
    decoder = VariationalDecoder(len(vocab), emb_dim, hidden_size,
                                 params["num_layers"],
                                 dropout_rate=params["dropout"],
                                 emb_matrix=emb_matrix)
    decoder.to(DEVICE)
    sos_idx = word2idx[SOS]
    eos_idx = word2idx[EOS]
    model = VariationalSeq2Seq(encoder, decoder, params["latent_dimension"],
                               sos_idx, eos_idx)
    logging.info(model)
    model.to(DEVICE)
    opt = torch.optim.Adam(model.trainable_parameters(),
                           lr=params["learn_rate"])
    loss_fn = vae_loss

    # If there is a checkpoint at checkpoint_dir, we load it and continue
    # training/evaluating from there.
    # If no checkpoints exist at checkpoint_dir, load_latest_checkpoint
    # will return the same model and opt, and start_epoch=0
    checkpoint_found = False
    logging.info("Trying to load latest model checkpoint from")
    logging.info(f"  {ckpt_dir}")
    model, opt, start_epoch, ckpt_fname = load_latest_checkpoint(
            model, opt, ckpt_dir)
    if ckpt_fname is None:
        logging.warning("No checkpoint found!")
    else:
        checkpoint_found = True
        logging.info(f"Loaded checkpoint '{ckpt_fname}'")

    # TRAINING LOOP
    if params["train"] is True:
        logging.info("TRAINING")
        logging.info("Ctrl-C to interrupt and save most recent model.")
        if checkpoint_found is False:
            logging.warning("No checkpoint found! Training from base model.")

        # Train the model for the specified number of epochs
        epoch_range = range(start_epoch, start_epoch + params["epochs"])
        for epoch in epoch_range:
            try:
                if epoch == 0:
                    # Log reconstructions before model training
                    log_reconstructions(model, train_data, idx2word, "train",
                                        -1, logdir)
                # Run a single training loop
                model, opt = train(
                        model, opt, train_dataloader, loss_fn, epoch, params,
                        idx2word, verbose=verbose, summary_writer=train_writer)
                # Log inputs and their reconstructions
                log_reconstructions(model, train_data, idx2word, "train",
                                    epoch, logdir)
                # Log active units
                log_z_metadata(model, train_dataloader, "train",
                               epoch, logdir, train_writer)
                # Measure how well the model autoencodes
                measure_autoencoding(model, train_dataloader,
                                     epoch, train_writer)

                if params["validate"] is True:
                    # Run a dev evaluation loop
                    evaluate(model, dev_dataloader, loss_fn, epoch, idx2word,
                             name="dev", verbose=verbose,
                             summary_writer=dev_writer)
                    # Log inputs and their reconstructions
                    log_reconstructions(model, dev_data, idx2word, "dev",
                                        epoch, logdir)
                    # Log active units
                    log_z_metadata(model, dev_dataloader, "dev",
                                   epoch, logdir, dev_writer)
                    # Measure how well the model autoencodes
                    measure_autoencoding(model, dev_dataloader,
                                         epoch, dev_writer)
            except KeyboardInterrupt:
                break

        # Save the model
        ckpt_fname = f"model_{epoch}.pt"
        ckpt_path = os.path.join(ckpt_dir, ckpt_fname)
        logging.info(f"Saving trained model to {ckpt_path}")
        torch.save({"model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "epoch": epoch},
                   ckpt_path)
        checkpoint_found = True
        start_epoch = epoch

    # VALIDATION LOOP
    if params["validate"] is True:
        logging.info("VALIDATING")
        if checkpoint_found is False:
            logging.warning("No checkpoint found! Evaluating untrained model!")
        evaluate(model, dev_dataloader, loss_fn, start_epoch, idx2word,
                 name="dev", verbose=verbose, summary_writer=None)
        # Log inputs and their reconstructions
        log_reconstructions(model, dev_data, idx2word, "dev",
                            start_epoch, logdir, n=30)
        # Log active units
        log_z_metadata(model, dev_dataloader, "dev",
                       start_epoch, logdir, dev_writer)

    # TESTING LOOP
    if params["test"] is True:
        logging.info("TESTING")
        if checkpoint_found is False:
            logging.warning("No checkpoint found! Evaluating untrained model!")
        evaluate(model, test_dataloader, loss_fn, start_epoch, idx2word,
                 name="test", verbose=verbose, summary_writer=None)

    if params["sample"] is True:
        logging.info("SAMPLING")
        if checkpoint_found is False:
            logging.warning("No checkpoint found! Sampling untrained model!")
        idx2word = {idx: word for (word, idx) in word2idx.items()}
        model.eval()

        for inputs in train_data:
            inputs = inputs.squeeze(0)
            in_text = [idx2word[w.item()] for w in inputs.squeeze()]
            encoded, context = model.encode(inputs.to(DEVICE), [len(in_text)])
            z_orig, mu, logvar = model.compute_z(context)
            mu = mu.detach().cpu()
            logvar = logvar.detach().cpu()

            orig_idxs = model.sample(z_orig)
            orig_predicted = [idx2word[idx] for idx in orig_idxs]
            for dim in range(model.latent_dim):
                print(dim)
                print(f"inp: '{' '.join(orig_predicted)}'")
                print(z_orig[0, dim])
                print("=======================")
                try:
                    start = -4
                    stop = 4
                    for z_i in np.linspace(start, stop, 30):
                        z_copy = z_orig.clone().detach()
                        z_copy[0, dim] = z_i
                        idxs = model.sample(z_copy)
                        predicted = [idx2word[idx] for idx in idxs]
                        polarity = '-' if 'not' in predicted else '+'
                        print(f"{polarity}:{len(predicted)}\t{z_i:.4f}\t\t{' '.join(predicted)}")  # noqa
                    input()
                except KeyboardInterrupt:
                    return

    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H:%M:%S")
    logging.info(f"END: {now_str}")


if __name__ == "__main__":
    args = parse_args()
    run(args.params_json, verbose=args.verbose)
