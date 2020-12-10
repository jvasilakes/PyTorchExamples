import os
import json
import random
import pickle
import logging
import argparse
import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.tensorboard import SummaryWriter

"""
A very basic BiLSTM for token classification. Implemented for
the CONLL03 NER task.

See here for a guide to variable length sequences in PyTorch
https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e
"""

SEED = 10
logging.basicConfig(level=logging.INFO)


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


class BiLSTMForTokenClassification(nn.Module):
    """
    Embedding -> BiLSTM -> Linear

    :param numpy.ndarray embedding_matrix: token_index X num_dims matrix of
                                           word_embeddings.
    :param int hidden_size: The size of the output representation of the LSTM,
                            and the input representation of the linear layer.
    :param int output_dim: The number of output labels.
    """
    def __init__(self, embedding_matrix, hidden_size, output_dim):
        super(BiLSTMForTokenClassification, self,).__init__()
        self.vocab_size, self.emb_dim = embedding_matrix.shape
        self.emb_layer = nn.Embedding.from_pretrained(
                torch.Tensor(embedding_matrix))
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.recurrent = nn.LSTM(self.emb_dim, self.hidden_size // 2,
                                 1, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(self.hidden_size, self.output_dim)

    def forward(self, inputs, lengths):
        embedded = self.emb_layer(inputs)
        # Pack all distinct sequences in the batch into a single matrix.
        packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False)
        out, h_n = self.recurrent(packed)
        # Unpack the batch back into distinct sequences.
        out_unpacked, lengths_unpacked = nn.utils.rnn.pad_packed_sequence(
                out, batch_first=True)
        dropped = self.dropout(out_unpacked)
        logits = torch.squeeze(self.linear(dropped))
        return logits


class TokenClassificationDataset(torch.utils.data.Dataset):
    """
    A basic dataset for token classification.

    :param List(torch.Tensor) Xs: A list of LongTensors representing
                                  the input documents. The members of
                                  each LongTensor are indices of words
                                  in an embedding matrix.
    :param List(torch.Tensor) Ys: A list of LongTensors representing
                                  the ground-truth labels for each input
                                  document.
    """
    def __init__(self, docs, tags, word2idx, tag2idx):
        self.docs = docs
        self.tags = tags
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.Xs = [self.doc2tensor(doc) for doc in self.docs]
        self.Ys = [self.tags2tensor(doctags) for doctags in self.tags]
        assert len(self.Xs) == len(self.Ys)
        if "<UNK>" not in word2idx.keys():
            raise ValueError("word2idx must have an '<UNK>' entry.")
        if "<PAD>" not in word2idx.keys():
            raise ValueError("word2idx must have an '<PAD>' entry.")
        if "<PAD>" not in tag2idx.keys():
            raise ValueError("tag2idx must have an '<PAD>' entry.")

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

    def tags2tensor(self, tags):
        return torch.LongTensor([self.tag2idx[tag] for tag in tags])


def get_conll03(path):
    """
    Read the CONLL 03 dataset into docs and tags.

    :param str path: Path to the data file.
    :rtype: Tuple(List(List), List(List))
    """
    lines = open(path).readlines()
    docs = []
    tags = []
    curr_doc = []
    curr_tags = []
    for line in lines:
        line = line.strip()
        if line == "":
            continue
        elif line.startswith("-DOCSTART-"):
            if curr_doc != []:
                assert len(curr_doc) == len(curr_tags)
                docs.append(curr_doc)
                tags.append(curr_tags)
            curr_doc = []
            curr_tags = []
        else:
            tok, pos, con, tag = line.split()
            curr_doc.append(tok.lower())
            curr_tags.append(tag)

    assert len(docs) == len(tags)
    return docs, tags


def pad_sequence(batch):
    """
    Pad the sequence batch with 0 vectors for X
    and 0 for Y. Meant to be used as the value of
    the `collate_fn` argument to `torch.utils.data.DataLoader`.
    """
    seqs = [torch.squeeze(x[0]) for x in batch]
    seqs_padded = nn.utils.rnn.pad_sequence(
            seqs, batch_first=True, padding_value=0)
    lengths = [len(s) for s in seqs]
    labs = [torch.LongTensor(y[1]) for y in batch]
    labs_padded = nn.utils.rnn.pad_sequence(
            labs, batch_first=True, padding_value=0)
    return seqs_padded, lengths, labs_padded


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


def accuracy(y_hat, y):
    predicted = torch.argmax(y_hat, dim=1)
    return sum(y == predicted) / float(len(y))


def train(model, opt, loss_fn, dataloader, tag2idx, epoch,
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
    num_tags = len(tag2idx)
    losses = []
    if verbose is True:
        pbar = tqdm(total=len(dataloader))
    batch_num = 1
    for (Xbatch, lengths, Ybatch) in dataloader:
        Xbatch = Xbatch.cuda()
        Ybatch = Ybatch.cuda()
        preds = model(Xbatch, lengths)
        preds = preds.view(-1, num_tags)
        Ybatch = Ybatch.view(-1)

        pad_lab = tag2idx["<PAD>"]
        idxs = torch.where(Ybatch != pad_lab)
        Ybatch = Ybatch[idxs]
        preds = preds[idxs[0], :]

        batch_loss = loss_fn(preds, Ybatch)
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
    writer.add_histogram("train_losses", np.array(losses), epoch)
    writer.add_scalar("train_loss", np.mean(losses), epoch)
    writer.flush()
    return model, opt, losses


def evaluate(model, loss_fn, dataloader, tag2idx, train_epoch,
             verbose=True, experiment_name=None):
    # Defaults to current datetime.
    if experiment_name is not None:
        experiment_name = f"runs/{experiment_name}"
    writer = SummaryWriter(log_dir=experiment_name)

    model.eval()
    num_tags = len(tag2idx)
    precs = []
    recs = []
    fs = []
    accs = []
    losses = []
    if verbose is True:
        pbar = tqdm(total=len(dataloader))
    for (Xbatch, lengths, Ybatch) in dataloader:
        Xbatch = Xbatch.cuda()
        Ybatch = Ybatch.cuda()
        preds = model(Xbatch, lengths)
        preds = preds.view(-1, num_tags)
        Ybatch = Ybatch.view(-1)
        pad_lab = tag2idx["<PAD>"]
        idxs = torch.where(Ybatch != pad_lab)
        Ybatch = Ybatch[idxs]
        preds = preds[idxs[0], :]
        batch_loss = loss_fn(preds, Ybatch)
        losses.append(batch_loss.detach().cpu().numpy())
        prec, rec, f1, _ = precision_recall_fscore_support(
                Ybatch.detach().cpu(),
                torch.argmax(preds, dim=1).detach().cpu(),
                average="macro", zero_division=0)
        precs.append(prec)
        recs.append(rec)
        fs.append(f1)
        acc = accuracy(preds, Ybatch).detach().cpu().numpy()
        accs.append(acc)
        if verbose is True:
            pbar.set_description(f"A: {acc:.2f}, P: {prec:.2f}, R: {rec:.2f}")
            pbar.update(1)
    if verbose is True:
        pbar.close()
    writer.add_histogram("eval_losses", np.array(losses), train_epoch)
    writer.add_scalar("eval_loss", np.mean(losses), train_epoch)
    writer.flush()
    logging.info(f"Loss: {np.mean(losses):.4f} +/- {np.std(losses):.4f}")
    logging.info(f"Accuracy: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    logging.info(f"Precision: {np.mean(precs):.4f} +/- {np.std(precs):.4f}")
    logging.info(f"Recall: {np.mean(recs):.4f} +/- {np.std(recs):.4f}")
    logging.info(f"F1: {np.mean(fs):.4f} +/- {np.std(fs):.4f}")


def validate_params(params):
    valid_params = {
            "name": str,
            "data_dir": str,
            "glove_path": str,
            "checkpoint_dir": str,
            "epochs": int,
            "batch_size": int,
            "learn_rate": float,
            "train": bool,
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
    params = json.load(open(params_file, 'r'))
    validate_params(params)
    logging.info("PARAMETERS:")
    for (param, val) in params.items():
        logging.info(f"  {param}: {val}")

    ckpt_dir = os.path.join(params["checkpoint_dir"], params["name"])
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    train_data_path = os.path.join(params["data_dir"], "train.txt")

    # Get training data.
    train_docs, train_tags = get_conll03(train_data_path)

    # Get the vocabulary and load the glove embeddings.
    vocab = ["<PAD>", "<UNK>"] + \
        list(sorted({word for doc in train_docs for word in doc}))
    logging.info(f"Loading embeddings from {params['glove_path']}")
    glove, emb_dim = load_glove(params["glove_path"])
    emb_matrix, word2idx = get_embedding_matrix(vocab, glove)

    uniq_tags = ["<PAD>"] + \
        list(sorted({t for doctags in train_tags for t in doctags}))
    tag2idx = {tag: idx for (idx, tag) in enumerate(uniq_tags)}

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # The hidden size is the power of 2 that is closest to, and greater
    # than, the embedding dimension.
    hidden_size = int(2**np.ceil(np.log2(emb_dim)))
    model = BiLSTMForTokenClassification(emb_matrix, hidden_size,
                                         len(uniq_tags))
    logging.info(model)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=params["learn_rate"])
    loss_fn = nn.CrossEntropyLoss()

    # If there is a checkpoint at checkpoint_dir, we load it and continue
    # training from there.
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

    if params["train"] is True:
        # Load the training dataset.
        train_conll03 = TokenClassificationDataset(train_docs, train_tags,
                                                   word2idx, tag2idx)
        train_dataloader = torch.utils.data.DataLoader(
                train_conll03, shuffle=True, batch_size=params["batch_size"],
                collate_fn=pad_sequence)
        if params["validate"] is True:
            valid_data_path = os.path.join(params["data_dir"], "valid.txt")
            dev_docs, dev_tags = get_conll03(valid_data_path)
            dev_conll03 = TokenClassificationDataset(
                    dev_docs, dev_tags, word2idx, tag2idx)
            dev_dataloader = torch.utils.data.DataLoader(
                    dev_conll03, shuffle=False,
                    batch_size=params["batch_size"],
                    collate_fn=pad_sequence)

        # Train the model
        logging.info("TRAINING")
        logging.info("Ctrl-C to interrupt and save most recent model.")
        if checkpoint_found is False:
            logging.warning("No checkpoint found! Training from base model.")
        epoch_range = range(start_epoch, start_epoch + params["epochs"])
        for epoch in epoch_range:
            try:
                model, opt, losses = train(
                        model, opt, loss_fn, train_dataloader,
                        tag2idx, epoch, verbose=verbose,
                        experiment_name=params["name"] + "/train")
                if params["validate"] is True:
                    evaluate(model, loss_fn, dev_dataloader, tag2idx,
                             epoch, verbose=verbose,
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

    if params["validate"] is True:
        logging.info("VALIDATING")
        if checkpoint_found is False:
            logging.warning("No checkpoint found! Evaluating untrained model!")
        valid_data_path = os.path.join(params["data_dir"], "valid.txt")
        dev_docs, dev_tags = get_conll03(valid_data_path)
        dev_conll03 = TokenClassificationDataset(
                dev_docs, dev_tags, word2idx, tag2idx)
        dev_dataloader = torch.utils.data.DataLoader(
                dev_conll03, shuffle=False, batch_size=params["batch_size"],
                collate_fn=pad_sequence)
        evaluate(model, loss_fn, dev_dataloader, tag2idx, start_epoch,
                 verbose=verbose, experiment_name=params["name"] + "/val")

    if params["test"] is True:
        logging.info("TESTING")
        if checkpoint_found is False:
            logging.warning("No checkpoint found! Evaluating untrained model!")
        test_data_path = os.path.join(params["data_dir"], "test.txt")
        test_docs, test_tags = get_conll03(test_data_path)
        test_conll03 = TokenClassificationDataset(
                test_docs, test_tags, word2idx, tag2idx)
        test_dataloader = torch.utils.data.DataLoader(
                test_conll03, shuffle=False, batch_size=params["batch_size"],
                collate_fn=pad_sequence)
        evaluate(model, loss_fn, test_dataloader, tag2idx, start_epoch,
                 verbose=verbose, experiment_name=params["name"] + "/test")


if __name__ == "__main__":
    set_seed(SEED)
    args = parse_args()
    run(args.params_json, verbose=args.verbose)
