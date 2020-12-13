import json
import argparse
import random

SEED = 10
random.seed(SEED)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str,
                        help="data/negated_pairs.jsonl")
    parser.add_argument("outfile", type=str,
                        help="Where to save the output.")
    return parser.parse_args()


def main(args):
    pairs = [json.loads(line) for line in open(args.infile, 'r')]
    masked_words = [pair["masked"] for pair in pairs]

    outlines = []
    for pair in pairs:
        pos, neg = process_pair(pair, masked_words)
        outlines.extend([pos, neg])

    with open(args.outfile, 'w') as outF:
        for line in outlines:
            json.dump(line, outF)
            outF.write('\n')


def process_pair(pair, all_masked_words):
    mask_tok = "[MASK]"
    pos_text = pair["positive"]
    neg_text = pair["negative"]
    pos_masked = pair["masked"]
    neg_masked = None
    while neg_masked is None:
        tmp = random.choice(all_masked_words)
        if tmp != pos_masked:
            neg_masked = tmp
    pos_text = pos_text.replace(mask_tok, pos_masked)
    neg_text = neg_text.replace(mask_tok, neg_masked)
    predicate = pair["predicateType"]
    pos = {"sentence": pos_text,
           "polarity": 1,
           "predicate": predicate}
    neg = {"sentence": neg_text,
           "polarity": 0,
           "predicate": predicate}
    return pos, neg


if __name__ == "__main__":
    args = parse_args()
    main(args)
