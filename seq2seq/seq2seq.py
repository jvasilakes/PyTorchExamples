import os
import re
import json
import random
import logging
import argparse
import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

"""
A basic LSTM Seq2Seq model for translation tasks.
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


class Encoder(nn.Module):
    """
    embedding -> dropout -> LSTM
    """
    def __init__(self, emb_dim, hidden_size, num_layers,
                 vocab_size, dropout_rate=0.5):
        super(Encoder, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.dropout_rate = dropout_rate
        self.embedding = nn.Embedding(self.vocab_size, self.emb_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.recurrent = nn.LSTM(self.emb_dim, self.hidden_size,
                                 num_layers=self.num_layers,
                                 dropout=self.dropout_rate, batch_first=True)

    def forward(self, inputs, hidden):
        # inputs: [batch_size, len(inputs)]
        embedded = self.dropout(self.embedding(inputs))
        # embedded: [batch_size, len(inputs), self.emb_dim]
        encoded, hidden = self.recurrent(embedded, hidden)
        # encoded: [batch_size, len(inputs), self.hidden_size]
        return encoded, hidden

    def init_hidden(self):
        # Initialize the LSTM state.
        # One for hidden and one for the cell
        return (torch.zeros(self.num_layers, 1,
                            self.hidden_size, device=DEVICE),
                torch.zeros(self.num_layers, 1,
                            self.hidden_size, device=DEVICE))


class Decoder(nn.Module):
    """
    embedding -> dropout -> LSTM -> linear
    """
    def __init__(self, emb_dim, hidden_size, num_layers,
                 vocab_size, dropout_rate=0.5):
        super(Decoder, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.dropout_rate = dropout_rate
        self.embedding = nn.Embedding(self.vocab_size, self.emb_dim)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.recurrent = nn.LSTM(self.emb_dim, self.hidden_size,
                                 num_layers=self.num_layers,
                                 dropout=self.dropout_rate, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, inputs, hidden):
        # inputs: [batch_size, len(inputs)]
        embedded = self.dropout(self.embedding(inputs))
        # embedded: [batch_size, len(inputs), self.emb_dim]
        decoded, hidden = self.recurrent(embedded, hidden)
        # decoded: [batch_size, len(inputs), self.hidden_size]
        logits = self.linear(decoded)
        # logits: [batch_size, len(inputs), self.vocab_size]
        logits = torch.squeeze(logits, 0)  # Drop the batch dimension.
        # logits: [len(inputs), self.vocab_size]
        return logits, hidden


class Seq2Seq(nn.Module):
    """
    LSTM_encoder -> final_state -> LSTM_decoder
    """
    def __init__(self, encoder, decoder, sos_token_idx, eos_token_idx):
        super(Seq2Seq, self,).__init__()
        assert isinstance(encoder, Encoder)
        assert isinstance(decoder, Decoder)
        self.encoder = encoder
        self.decoder = decoder
        self.sos_token_idx = sos_token_idx
        self.eos_token_idx = eos_token_idx

    def forward(self, inputs, outputs=None, teacher_forcing_prob=0.5):
        """
        Run a single example through the model.
        Set outputs=None, teacher_forcing_prob = 0.0 to run prediction
        """
        encoder_hidden = self.encoder.init_hidden()
        encoded, encoder_hidden = self.encoder(inputs, encoder_hidden)

        if outputs is not None:
            target_length = outputs.size(-1)
        else:
            target_length = 2 * inputs.size(-1)

        decoder_input = torch.LongTensor([[self.sos_token_idx]]).to(DEVICE)
        # Reshape BiLSTM hidden_size // 2 into hidden_size for LSTM.
        # decoder_hidden = [h.view(-1, 1, self.decoder.hidden_size)
        #                   for h in encoder_hidden]
        decoder_hidden = encoder_hidden

        vocab_size = self.decoder.vocab_size
        decoder_outputs = torch.zeros(target_length, vocab_size).to(DEVICE)
        for i in range(target_length):
            logits, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden)
            decoder_outputs[i] = logits
            use_teacher_forcing = random.random() < teacher_forcing_prob
            if use_teacher_forcing is True:
                # We lose a dimension slicing outputs, so we add it back.
                target = outputs[:, i]
                decoder_input = torch.unsqueeze(target, 0)
            else:
                decoder_input = logits.argmax(1).detach()
                if teacher_forcing_prob == 0.0:
                    if decoder_input.item() == self.eos_token_idx:
                        decoder_outputs = decoder_outputs[:i+1, :]
                        break
                decoder_input = torch.unsqueeze(decoder_input, 0)
        return decoder_outputs


class TranslationDataset(torch.utils.data.Dataset):
    """
    A basic dataset for token classification.

    :param List(list) indocs: A list of lists representing
                              the input documents. The members of
                              each list are indices of words
                              in an embedding matrix.
    :param List(list) outdocs: A list of lists representing
                               the ground-truth translation of indocs.
                               The members of each list are indices
                               of words in an embedding matrix.
    :param dict in_word2idx: A dictionary of tokens to integer indices
                             corresponding to an embedding matrix.
    :param dict out_word2idx: A dictionary of tokens to integer indices
                              corresponding to an embedding matrix.
    """
    def __init__(self, indocs, outdocs, in_word2idx, out_word2idx):
        if "<UNK>" not in in_word2idx.keys():
            raise ValueError("in_word2idx must have an '<UNK>' entry.")
        if "<UNK>" not in out_word2idx.keys():
            raise ValueError("out_word2idx must have an '<UNK>' entry.")
        if "<PAD>" not in in_word2idx.keys():
            raise ValueError("in_word2idx must have an '<PAD>' entry.")
        if "<PAD>" not in out_word2idx.keys():
            raise ValueError("out_word2idx must have an '<PAD>' entry.")
        self.indocs = indocs
        self.outdocs = outdocs
        self.in_word2idx = in_word2idx
        self.out_word2idx = out_word2idx
        self.Xs = [self.doc2tensor(doc, self.in_word2idx)
                   for doc in self.indocs]
        self.Ys = [self.doc2tensor(doc, self.out_word2idx)
                   for doc in self.outdocs]
        assert len(self.Xs) == len(self.Ys)

    def __getitem__(self, idx):
        return (self.Xs[idx], self.Ys[idx])

    def __len__(self):
        return len(self.Xs)

    def doc2tensor(self, doc, word2idx):
        idxs = []
        for w in doc:
            try:
                idxs.append(word2idx[w])
            except KeyError:
                idxs.append(word2idx["<UNK>"])
        return torch.LongTensor([idxs])


def get_translation_data(path):
    """
    Read the tab-separated translation data into inputs and outputs.

    :param str path: Path to the data file.
    :rtype: Tuple(List(List), List(List))
    """
    eng_fra = [pair.strip().lower().split('\t') for pair in open(path, 'r')]
    # We expect a 2-tuple for each example.
    assert {len(x) for x in eng_fra} == {2}
    # Experimenting with translating both ways in the same model.
    # eng_fra = eng_fra + [[p[1], p[0]] for p in eng_fra]
    # random.shuffle(eng_fra)
    inputs = [x[0] for x in eng_fra]
    outputs = [x[1] for x in eng_fra]
    return inputs, outputs


def preprocess(text_data, SOS, EOS, reverse=False):
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
    for doc in text_data:
        doc = doc.lower().strip()
        doc = re.sub(r"([.!?])", r" \1", doc)
        doc = re.sub(r"[^a-zA-Z.!?\[\]]+", r" ", doc)
        doc = doc.split()  # Tokenize
        if reverse is True:
            doc = doc[::-1]
        doc = [SOS] + doc + [EOS]
        out_data.append(doc)
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


def train(model, opt, dataset, loss_fn, epoch,
          grad_accumulation_steps=1, verbose=True,
          experiment_name=None):
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
    avg_loss = 0.0
    if verbose is True:
        pbar = tqdm(total=len(dataset))
    for (step, (X, Y)) in enumerate(dataset):
        X = X.to(DEVICE)
        Y = Y.to(DEVICE)
        outputs = model(X, Y)
        loss = loss_fn(outputs, torch.squeeze(Y)) / grad_accumulation_steps
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        if step % grad_accumulation_steps == 0:
            opt.step()
            opt.zero_grad()
        losses.append(loss.detach().cpu().numpy())
        if verbose is True:
            pbar.update(1)
            pbar.set_description(f"Loss: {np.mean(losses):.4f}")
    opt.step()
    opt.zero_grad()
    if verbose is True:
        pbar.close()

    writer.add_histogram("train_losses", np.array(losses), epoch)
    writer.add_scalar("train_loss", np.mean(losses), epoch)
    writer.flush()
    logging.info(f"({epoch}) Train loss: {np.mean(losses):.4f} +/- {np.std(losses):.4f}")  # noqa
    return model, opt, avg_loss


def evaluate(model, opt, dataset, loss_fn, train_epoch,
             verbose=True, experiment_name=None):
    raise NotImplementedError("evaluate() is still in development.")

    if experiment_name is not None:
        experiment_name = f"runs/{experiment_name}"
    writer = SummaryWriter(log_dir=experiment_name)

    model.eval()
    losses = []
    if verbose is True:
        pbar = tqdm(total=len(dataset))
    for (X, Y) in dataset:
        X = X.to(DEVICE)
        Y = Y.to(DEVICE)
        outputs = model(X, Y, teacher_forcing_prob=0.0)
        # TODO: pad outputs and Y to the same length so loss_fn doesn't fail.
        loss = loss_fn(outputs, torch.squeeze(Y)).detach().cpu().numpy()
        losses.append(loss)
        if verbose is True:
            pbar.update(1)
            pbar.set_description(f"Loss: {loss:.4f}")  # noqa
    if verbose is True:
        pbar.close()
    writer.add_histogram("train_losses", np.array(losses), train_epoch)
    writer.add_scalar("train_loss", np.mean(losses), train_epoch)
    writer.flush()
    logging.info(f"Eval loss: {np.mean(losses):.4f} +/- {np.std(losses):.4f}")


def validate_params(params):
    valid_params = {
            "name": str,
            "data_path": str,
            "checkpoint_dir": str,
            "reverse_input": bool,
            "embedding_dimension": int,
            "num_layers": int,
            "epochs": int,
            "batch_size": int,
            "grad_accumulation_steps": int,
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
    PAD = "<PAD>"
    UNK = "<UNK>"

    params = json.load(open(params_file, 'r'))
    validate_params(params)
    logging.info("PARAMETERS:")
    for (param, val) in params.items():
        logging.info(f"  {param}: {val}")

    ckpt_dir = os.path.join(params["checkpoint_dir"], params["name"])
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    # Get train / val / test splits
    logging.info("Splitting data")
    indocs, outdocs = get_translation_data(params["data_path"])
    indocs = preprocess(indocs, SOS, EOS, reverse=params["reverse_input"])
    outdocs = preprocess(outdocs, SOS, EOS, reverse=False)
    in_train, in_test, out_train, out_test = train_test_split(
            indocs, outdocs, test_size=0.25, shuffle=True)
    in_train, in_val, out_train, out_val = train_test_split(
            in_train, out_train, test_size=0.25, shuffle=True)
    logging.info(f"Training examples: {len(in_train)}")
    logging.info(f"Validation examples: {len(in_val)}")
    logging.info(f"Testing examples: {len(in_test)}")

    # Get the input and output vocabulary.
    in_vocab = [SOS, EOS, PAD, UNK] + \
        list(sorted({word for doc in in_train for word in doc}))
    in_word2idx = {word: idx for (idx, word) in enumerate(in_vocab)}
    out_vocab = [SOS, EOS, PAD, UNK] + \
        list(sorted({word for doc in out_train for word in doc}))
    out_word2idx = {word: idx for (idx, word) in enumerate(out_vocab)}

    # Load the datasets.
    train_data = TranslationDataset(in_train, out_train,
                                    in_word2idx, out_word2idx)
    val_data = TranslationDataset(in_val, out_val,
                                  in_word2idx, out_word2idx)
    test_data = TranslationDataset(in_test, out_test,
                                   in_word2idx, out_word2idx)

    # The hidden size is the power of 2 that is closest to, and greater
    # than, the embedding dimension.
    emb_dim = params["embedding_dimension"]
    hidden_size = int(2**np.ceil(np.log2(emb_dim+1)))
    encoder = Encoder(emb_dim, hidden_size,
                      params["num_layers"], len(in_vocab))
    encoder.to(DEVICE)
    decoder = Decoder(emb_dim, hidden_size,
                      params["num_layers"], len(out_vocab))
    decoder.to(DEVICE)

    sos_idx = out_word2idx[SOS]
    eos_idx = out_word2idx[EOS]

    model = Seq2Seq(encoder, decoder, sos_idx, eos_idx)
    logging.info(model)
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=params["learn_rate"])
    loss_fn = nn.CrossEntropyLoss()

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

    if params["train"] is True:
        logging.info("TRAINING")
        logging.info("Ctrl-C to interrupt and save most recent model.")
        if checkpoint_found is False:
            logging.warning("No checkpoint found! Training from base model.")
        # Train the model
        epoch_range = range(start_epoch, start_epoch + params["epochs"])
        for epoch in epoch_range:
            try:
                model, opt, losses = train(
                        model, opt, train_data, loss_fn, epoch,
                        grad_accumulation_steps=params["grad_accumulation_steps"],  # noqa
                        verbose=verbose, experiment_name=params["name"] + "/train")  # noqa
                if params["validate"] is True:
                    evaluate(model, opt, val_data, loss_fn, epoch,
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
        out_idx2word = {idx: word for (word, idx) in out_word2idx.items()}
        model.eval()
        # For some reason model.eval() doesn't change the encoder dropout.
        model.encoder.recurrent.dropout = 0.0
        sample_dataset = val_data
        sample_idxs = np.random.choice(len(sample_dataset),
                                       size=len(sample_dataset),
                                       replace=False)
        i = 0
        while True:
            sample_idx = sample_idxs[i]
            i += 1
            source_text = sample_dataset.indocs[sample_idx]
            target_text = sample_dataset.outdocs[sample_idx]
            source, target = sample_dataset[sample_idx]
            print(f"src: '{' '.join(source_text[::-1])}'")
            print(f"trg: '{' '.join(target_text)}'")
            logits = model(source.to(DEVICE), teacher_forcing_prob=0.0)
            pred_idxs = logits.argmax(dim=1)
            predicted = [out_idx2word[idx.item()]
                         for idx in pred_idxs if idx != 0]
            print(f"prd: '{' '.join(predicted)}'")
            input("[ENTER] to continue. Ctl-C to quit.")

    if params["validate"] is True:
        logging.info("VALIDATING")
        if checkpoint_found is False:
            logging.warning("No checkpoint found! Evaluating untrained model!")
        evaluate(model, opt, val_data, loss_fn, start_epoch,
                 verbose=verbose, experiment_name=params["name"] + "/val")

    if params["test"] is True:
        logging.info("TESTING")
        if checkpoint_found is False:
            logging.warning("No checkpoint found! Evaluating untrained model!")
        evaluate(model, opt, test_data, loss_fn, start_epoch,
                 verbose=verbose, experiment_name=params["name"] + "/val")


if __name__ == "__main__":
    set_seed(SEED)
    args = parse_args()
    run(args.params_json, verbose=args.verbose)
