import os
import re
import json
import random
import pickle
import logging
import argparse
import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

"""
A very basic BiLSTM for token classification. Implemented for
the CONLL03 NER task.

See here for a guide to variable length sequences in PyTorch
https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e
"""

SEED = 10
logging.basicConfig(level=logging.INFO)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("params_json", type=str,
                        help="""Path to JSON file containing experiment
                                parameters which has the following schema:
                                    {'data_dir': str,
                                     'glove_path': str,
                                     'epochs': int,
                                     'batch_size': int,
                                     'train': bool,
                                     'validate': bool,
                                     'test': bool}""")
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


class ConditionedRNNLM(nn.Module):
    """
    A basic LSTM language model which conditions the output
    on a vector of one or more attributes.
    """
    def __init__(self, embedding_matrix, label_size,
                 hidden_size, output_dim, dropout_rate=0.5):
        super(ConditionedRNNLM, self,).__init__()
        self.vocab_size, self.emb_dim = embedding_matrix.shape
        self.emb_layer = nn.Embedding.from_pretrained(
                torch.Tensor(embedding_matrix))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.label_size = label_size
        self.hidden_size = hidden_size
        self.num_layers = 2
        self.output_dim = output_dim
        self.recurrent = nn.LSTM(
                self.emb_dim + self.label_size, self.hidden_size,
                num_layers=self.num_layers, dropout=dropout_rate,
                batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_dim)

    def forward(self, inputs, lengths, label_vecs, hidden=None):
        embedded = self.emb_layer(inputs)
        conditioned = torch.cat([embedded, label_vecs], dim=2)
        # Pack all distinct sequences in the batch into a single matrix.
        packed = nn.utils.rnn.pack_padded_sequence(
                conditioned, lengths, batch_first=True, enforce_sorted=False)
        out, hidden = self.recurrent(packed, hidden)
        # Unpack the batch back into distinct sequences.
        out_unpacked, lengths_unpacked = nn.utils.rnn.pad_packed_sequence(
                out, batch_first=True)
        dropped = self.dropout(out_unpacked)
        logits = torch.squeeze(self.linear(dropped))
        return logits, hidden

    def sample(self, prompt, label_vec, eos_token_idx, max_length=15):
        hidden = self.init_hidden()
        logits, hidden = self.forward(prompt, [prompt.size(1)],
                                      label_vec, hidden)
        logits = torch.unsqueeze(logits, 0)
        next_input = logits.argmax(2)[:, -1]
        yield next_input.item()
        label_vec = torch.unsqueeze(label_vec[:, 0, :], 1)
        for i in range(max_length):
            if next_input == eos_token_idx:
                break
            logits, hidden = self.forward(
                    next_input[None, :], [1], label_vec, hidden)
            logits = torch.unsqueeze(logits, 0)
            next_input = logits.argmax(1).detach()
            yield next_input.item()

    def init_hidden(self):
        return (torch.zeros(self.num_layers, 1,
                            self.hidden_size, device=DEVICE),
                torch.zeros(self.num_layers, 1,
                            self.hidden_size, device=DEVICE))


class LabeledTextDataset(torch.utils.data.Dataset):
    """
    A basic dataset for conditional language modeling
    or sequence classification.
    """
    def __init__(self, docs, labels, word2idx):
        self.docs = docs
        self.labels = labels
        self.word2idx = word2idx
        self.Xs = [self.doc2tensor(doc) for doc in self.docs]
        self.Ys = [torch.LongTensor(labs) for labs in self.labels]
        assert len(self.Xs) == len(self.Ys)
        if "<UNK>" not in word2idx.keys():
            raise ValueError("word2idx must have an '<UNK>' entry.")
        if "<PAD>" not in word2idx.keys():
            raise ValueError("word2idx must have an '<PAD>' entry.")

    def __getitem__(self, idx):
        return (self.Xs[idx], self.Ys[idx])

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


def get_sentences_labels(path):
    """
    Read the output of get_data.py
    """
    sentences = []
    labels = []
    with open(path, 'r') as inF:
        for line in inF:
            data = json.loads(line)
            sentences.append(data["sentence"])
            labs = {"polarity": data["polarity"],
                    "predicate": data["predicate"]}
            labels.append(labs)
    return sentences, labels


def preprocess_sentences(sentences, SOS, EOS):
    """
    Preprocess the input text by
      lowercasing
      adding space around punctuation
      removing strange characters
      tokenizing on whitespace
      (optional) reversing the words as suggested by the Seq2Seq paper
      adding <SOS> and <EOS> tokens
    """
    out_data = []
    for doc in sentences:
        doc = doc.lower().strip()
        doc = re.sub(r"([.!?])", r" \1", doc)
        doc = re.sub(r"[^a-zA-Z.!?]+", r" ", doc)
        doc = doc.split()  # Tokenize
        doc = [SOS] + doc + [EOS]
        out_data.append(doc)
    return out_data


def preprocess_labels(labels):
    """
    Make a list of one-hot labels for the predicate
    and polarity of each sentence.
    """
    lb = LabelBinarizer()
    preds = [lab["predicate"] for lab in labels]
    onehot_preds = lb.fit_transform(preds)

    onehot_labels = [np.hstack([lab["polarity"], pred])
                     for (lab, pred) in zip(labels, onehot_preds)]
    return onehot_labels, lb


def pad_sequence(batch):
    """
    Pad the sequence batch with 0 vectors for X
    and 0 for Y. Meant to be used as the value of
    the `collate_fn` argument to `torch.utils.data.DataLoader`.
    """
    seqs = [torch.squeeze(x[0]) for x in batch]
    lengths = [len(s) for s in seqs]
    seqs_padded = nn.utils.rnn.pad_sequence(
            seqs, batch_first=True, padding_value=0)

    labels = [b[1] for b in batch]
    max_len = seqs_padded.size(1)
    labels_repeated = torch.cat([lab.repeat(1, max_len, 1) for lab in labels])
    return seqs_padded, lengths, labels_repeated


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
    pickle.dump(glove, open(pickle_file, "wb"))
    return glove, emb_dim


def get_embedding_matrix(vocab, glove):
    emb_dim = len(list(glove.values())[0])
    matrix = np.zeros((len(vocab), emb_dim))
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


def load_latest_checkpoint(model, optimizer, checkpoint_dir):
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


def train(model, opt, loss_fn, dataloader, epoch,
          verbose=True, experiment_name=None):
    """
    Perform a single training epoch on dataset.
    Save the train loss to a Tensorboard logdir at
    runs/{experiment_name}
    """

    if experiment_name is not None:
        experiment_name = f"runs/{experiment_name}"
    writer = SummaryWriter(log_dir=experiment_name)

    model.train()
    losses = []
    if verbose is True:
        pbar = tqdm(total=len(dataloader))
    batch_num = 1
    for (Xbatch, lengths, Ybatch) in dataloader:
        model.init_hidden()
        Xbatch = Xbatch.to(DEVICE)
        Ybatch = Ybatch.to(DEVICE)
        Xbatch_in = Xbatch[:, :-1]
        Xbatch_out = Xbatch[:, 1:]
        Ybatch_in = Ybatch[:, :-1, :]
        lengths = [length - 1 for length in lengths]
        preds, _ = model(Xbatch_in, lengths, Ybatch_in)
        Xbatch_out = torch.flatten(Xbatch_out)
        preds = preds.view(-1, model.output_dim)
        batch_loss = loss_fn(preds, Xbatch_out)
        batch_loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(batch_loss.detach().cpu().numpy())
        if verbose is True:
            pbar.update(1)
            pbar.set_description(f"({epoch}), Loss: {batch_loss:.4f}")
        batch_num += 1
    if verbose is True:
        pbar.close()
    logging.info(f"Avg. epoch loss: {np.mean(losses):.4f}")
    logging.info(f"Avg. epoch ppl: {perplexity(losses):.4f}")
    writer.add_histogram("train_losses", np.array(losses), epoch)
    writer.add_scalar("train_loss", np.mean(losses), epoch)
    writer.add_scalar("ppl", perplexity(losses), epoch)
    writer.flush()
    return model, opt, losses


def perplexity(losses):
    return 2**(np.mean(losses))


def evaluate(model, loss_fn, dataloader, train_epoch,
             verbose=True, experiment_name=None):
    raise NotImplementedError("Evaluation not yet implemented.")


def validate_params(params):
    valid_params = {
            "name": str,
            "data_path": str,
            "glove_path": str,
            "checkpoint_dir": str,
            "epochs": int,
            "batch_size": int,
            "learn_rate": float,
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
    logging.info("PARAMETERS:")
    for (param, val) in params.items():
        logging.info(f"  {param}: {val}")

    ckpt_dir = os.path.join(params["checkpoint_dir"], params["name"])
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    sentences, labels = get_sentences_labels(params["data_path"])
    sentences = preprocess_sentences(sentences, SOS, EOS)
    labels, label_binarizer = preprocess_labels(labels)

    train_sents, test_sents, train_labs, test_labs = train_test_split(
            sentences, labels, test_size=0.2,
            shuffle=True, stratify=labels)
    train_sents, val_sents, train_labs, val_labs = train_test_split(
            train_sents, train_labs, test_size=0.2,
            shuffle=True, stratify=train_labs)

    # Get the vocabulary and load the glove embeddings.
    vocab = ["<PAD>", "<UNK>"] + \
        list(sorted({word for doc in train_sents for word in doc}))
    logging.info(f"Loading embeddings from {params['glove_path']}")
    glove, emb_dim = load_glove(params["glove_path"])
    emb_matrix, word2idx = get_embedding_matrix(vocab, glove)

    if params["train"] is True:
        # Load the training dataset.
        train_data = LabeledTextDataset(
                train_sents, train_labs, word2idx)
        train_dataloader = torch.utils.data.DataLoader(
                train_data, shuffle=True, batch_size=params["batch_size"],
                collate_fn=pad_sequence)
        logging.info(f"Training examples: {len(train_sents)}")
    if params["validate"] is True:
        val_data = LabeledTextDataset(
                val_sents, val_labs, word2idx)
        val_dataloader = torch.utils.data.DataLoader(
                val_data, shuffle=False,
                batch_size=params["batch_size"],
                collate_fn=pad_sequence)
        logging.info(f"Validation examples: {len(val_sents)}")
    if params["test"] is True:
        test_data = LabeledTextDataset(
                test_sents, test_labs, word2idx)
        test_dataloader = torch.utils.data.DataLoader(
                test_data, shuffle=False,
                batch_size=params["batch_size"],
                collate_fn=pad_sequence)
        logging.info(f"Testing examples: {len(test_sents)}")

    # The hidden size is the power of 2 that is closest to, and greater
    # than, the embedding dimension.
    label_size = train_labs[0].shape[0]
    hidden_size = int(2**np.ceil(np.log2(emb_dim + label_size)))
    model = ConditionedRNNLM(emb_matrix, label_size,
                             hidden_size, len(vocab))
    logging.info(model)
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=params["learn_rate"])
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    # If there is a checkpoint at checkpoint_dir, we load it and continue
    # training from there.
    # If no checkpoints exist at checkpoint_dir, load_latest_checkpoint
    # will return the same model and opt, and epoch=0
    checkpoint_found = False
    logging.info("Trying to load latest model checkpoint from")
    logging.info(f"  {ckpt_dir}")
    model, opt, epoch, ckpt_fname = load_latest_checkpoint(
            model, opt, ckpt_dir)
    if ckpt_fname is None:
        logging.warning("No checkpoint found!")
    else:
        checkpoint_found = True
        logging.info(f"Loaded checkpoint '{ckpt_fname}'")

    if params["train"] is True:
        # Train the model
        logging.info("TRAINING")
        logging.info("Ctrl-C to interrupt and save most recent model.")
        if checkpoint_found is False:
            logging.warning("No checkpoint found! Training from base model.")
        epoch_range = range(epoch, epoch + params["epochs"])
        for epoch in epoch_range:
            try:
                model, opt, losses = train(
                        model, opt, loss_fn, train_dataloader,
                        epoch, verbose=verbose,
                        experiment_name=params["name"] + "/train")
                if params["validate"] is True:
                    evaluate(model, loss_fn, val_dataloader, epoch,
                             verbose=verbose,
                             experiment_name=params["name"] + "/val")
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

    if params["sample"] is True:
        logging.info("SAMPLING")
        if checkpoint_found is False:
            logging.warning("No checkpoint found! Sampling untrained model!")

        idx2word = {idx: word for (word, idx) in word2idx.items()}

        model.eval()
        for predicate in label_binarizer.classes_:
            for polarity in [1, 0]:
                polarity_t = torch.tensor([polarity])
                pred_onehot = torch.tensor(
                        label_binarizer.transform([predicate]))
                label_vec = torch.cat([polarity_t, pred_onehot[0]])
                prompt_words = [SOS, "a", "bowl"]
                prompt = torch.LongTensor([[word2idx[word]
                                            for word in prompt_words]])
                label_vecs = torch.tensor(
                        label_vec).repeat(1, len(prompt_words), 1)
                word_idxs = model.sample(prompt.to(DEVICE),
                                         label_vecs.to(DEVICE),
                                         word2idx[EOS])
                print(f"({polarity}) : {predicate}")
                words = prompt_words + [idx2word[idx] for idx in word_idxs]
                print(' '.join(words))
                input()

    if params["validate"] is True:
        logging.info("VALIDATING")
        if checkpoint_found is False:
            logging.warning("No checkpoint found! Evaluating untrained model!")
        evaluate(model, loss_fn, val_dataloader, epoch,
                 verbose=verbose, experiment_name=params["name"] + "/val")

    if params["test"] is True:
        logging.info("TESTING")
        if checkpoint_found is False:
            logging.warning("No checkpoint found! Evaluating untrained model!")
        evaluate(model, loss_fn, test_dataloader, epoch,
                 verbose=verbose, experiment_name=params["name"] + "/test")


if __name__ == "__main__":
    set_seed(SEED)
    args = parse_args()
    run(args.params_json, verbose=args.verbose)
