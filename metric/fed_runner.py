import argparse

from tqdm import tqdm

import metric.fed as fed
from utils import load_json, report_correlation

model, tokenizer = fed.load_models('microsoft/DialoGPT-large')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="Paths of the test data.")

    args = parser.parse_args()
    return args


def interactive_test(delimiter):
    while True:
        history = input('History:')
        if history == '':
            break
        utterance = input('Utt:')
        dialogue = history.split(delimiter)
        dialogue.append(utterance)
        dialogue = '<|endoftext|>' + ' <|endoftext|> '.join(dialogue)
        score = fed.evaluate(dialogue, model, tokenizer)
        print(score)


def batch_test(delimiter, file_dir):
    data = load_json(file_dir)
    gt_scores = []
    fed_correct, fed_semantic, fed_understandable = [], [], []
    for sample in tqdm(data):
        history = sample['history']
        utterance = sample['response']
        gt_scores.append(sample['event_cs'])
        dialogue = history.split(delimiter)
        if dialogue[-1] == '':
            dialogue.pop()
        dialogue.append(utterance)
        dialogue = '<|endoftext|>' + ' <|endoftext|> '.join(dialogue)
        score = fed.evaluate(dialogue, model, tokenizer)
        fed_correct.append(score['correct'])
        fed_semantic.append(score['semantically appropriate'])
        fed_understandable.append(score['understandable'])

    # Calculate metrics.
    scores = [fed_semantic, fed_understandable]
    names = ['FED semantically appropriate', 'FED understandable']

    for auto_score, name in zip(scores, names):
        print(name)
        report_correlation(auto_score, gt_scores)


if __name__ == '__main__':
    # interactive_test(delimiter='[SEP]')
    args = get_args()
    batch_test('</UTT>', args.data_dir)
