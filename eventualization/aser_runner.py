import argparse

import os
import pandas as pd
import warnings
from tqdm import tqdm

from eventualization.aser.extract.aser_extractor import SeedRuleASERExtractorForConversation
from eventualization.evt_evaluate import RELATIONS, compute_metrics
from utils import load_json

warnings.filterwarnings("ignore")


def replace_pronoun(head, tail, single):
    head_words = head.split()
    tail_words = tail.split()
    head_words_modified = []
    tail_words_modified = []
    if 'i' in tail_words or 'I' in tail_words:
        for word in tail_words:
            if word == 'i' or word == 'I':
                word = 'PersonX'
            if word == 'my' or word == 'My':
                word = "PersonX's"
            if word == 'you' or word == 'You':
                word = 'PersonY'
            if word == 'your' or word == 'Your':
                word = "PersonY's"
            tail_words_modified.append(word)
        for word in head_words:
            if single:  # The speaker doesn't change.
                if word == 'i' or word == 'I':
                    word = 'PersonX'
                if word == 'my' or word == 'My':
                    word = "PersonX's"
                if word == 'you' or word == 'You':
                    word = 'PersonY'
                if word == 'your' or word == 'Your':
                    word = "PersonY's"
                head_words_modified.append(word)
            else:  # The speaker changes.
                if word == 'i' or word == 'I':
                    word = 'PersonY'
                if word == 'my' or word == 'My':
                    word = "PersonY's"
                if word == 'you' or word == 'You':
                    word = 'PersonX'
                if word == 'your' or word == 'Your':
                    word = "PersonX's"
                head_words_modified.append(word)
    elif 'you' in tail_words or 'You' in tail_words:
        for word in tail_words:
            if word == 'i' or word == 'I':
                word = 'PersonY'
            if word == 'my' or word == 'My':
                word = "PersonY's"
            if word == 'you' or word == 'You':
                word = 'PersonX'
            if word == 'your' or word == 'Your':
                word = "PersonX's"
            tail_words_modified.append(word)
        for word in head_words:
            if single:  # The speaker doesn't change.
                if word == 'i' or word == 'I':
                    word = 'PersonY'
                if word == 'my' or word == 'My':
                    word = "PersonY's"
                if word == 'you' or word == 'You':
                    word = 'PersonX'
                if word == 'your' or word == 'Your':
                    word = "PersonX's"
                head_words_modified.append(word)
            else:  # The speaker changes.
                if word == 'i' or word == 'I':
                    word = 'PersonX'
                if word == 'my' or word == 'My':
                    word = "PersonX's"
                if word == 'you' or word == 'You':
                    word = 'PersonY'
                if word == 'your' or word == 'Your':
                    word = "PersonY's"
                head_words_modified.append(word)
    else:
        head_words_modified = head_words
        tail_words_modified = tail_words

    # Check grammar error.
    for i in range(len(head_words_modified)):
        if head_words_modified[i] == "'m" or head_words_modified[i] == "'re":
            head_words_modified[i] = 'be'
    for i in range(len(tail_words_modified)):
        if tail_words_modified[i] == "'m" or tail_words_modified[i] == "'re":
            tail_words_modified[i] = 'be'

    head_modified = ' '.join(head_words_modified)
    tail_modified = ' '.join(tail_words_modified)
    # Fix trivial typos.
    head_modified = head_modified.replace('PersonX m ', 'PersonX be ')
    head_modified = head_modified.replace('PersonY m ', 'PersonY be ')
    head_modified = head_modified.replace('PersonX re ', 'PersonX be ')
    head_modified = head_modified.replace('PersonY re ', 'PersonY be ')
    tail_modified = tail_modified.replace('PersonX m ', 'PersonX be ')
    tail_modified = tail_modified.replace('PersonY m ', 'PersonY be ')
    tail_modified = tail_modified.replace('PersonX re ', 'PersonX be ')
    tail_modified = tail_modified.replace('PersonY re ', 'PersonY be ')

    return head_modified, tail_modified


def normalize_evt(head, tail, relation, single):
    """Normalize the personal pronoun in head and tail to PersonX / PersonY."""
    head_modified, tail_modified = replace_pronoun(head, tail, single)
    # Fix o relations.
    if relation == 'oWant' or relation == 'oEffect' or relation == 'oReact':
        if tail_modified[:7] == 'PersonX':
            # Swap 'PersonX' and 'PersonY'.
            head_modified = head_modified.replace('PersonX', 'PersonZ')
            head_modified = head_modified.replace('PersonY', 'PersonX')
            head_modified = head_modified.replace('PersonZ', 'PersonY')
            tail_modified = tail_modified.replace('PersonX', 'PersonZ')
            tail_modified = tail_modified.replace('PersonY', 'PersonX')
            tail_modified = tail_modified.replace('PersonZ', 'PersonY')

    return head_modified, tail_modified


def aser_evt(contexts, utterances, threshold=0.8, stanfordnlp_dir='stanford-corenlp-4.5.0',
             relation_classifier_dir='models/aser_classifiers'):
    os.environ['CORENLP_HOME'] = stanfordnlp_dir
    aser_extractor = SeedRuleASERExtractorForConversation(
        corenlp_path=stanfordnlp_dir,
        corenlp_port=9000,
        relation_classifier_dir=relation_classifier_dir
    )

    outputs = []
    for i in tqdm(range(len(contexts))):
        prev_utt = contexts[i].split('</UTT>')[-2]
        utt = utterances[i]
        evt_output = aser_extractor.extract_relations_from_conversation(prev_utt, utt, threshold=threshold)
        refined_output = []
        for head, relation, tail, is_single in evt_output:
            normalized_head, normalized_tail = normalize_evt(head, tail, relation, is_single)
            refined_output.append((normalized_head, relation, normalized_tail))
        outputs.append(refined_output)

    return outputs


def benchmark():
    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--saved_tuple_path', type=str, help='Event-relation extraction results dir',
                            default='aser_extraction_result.csv')
        parser.add_argument('--data_dir', type=str, help='Test data dir.',
                            default='./data/deco/deco_test.json')
        parser.add_argument('--stanfordnlp_dir', type=str, help='Stanford core NLP dir.',
                            default='stanford-corenlp-4.5.0')
        parser.add_argument('--relation_classifier_dir', type=str, help='LSTM classifier for relation prediction.',
                            default='./models/aser_classifiers')
        return parser.parse_args()

    args = get_args()

    stanfordnlp_dir = args.stanfordnlp_dir
    relation_classifier_dir = args.relation_classifier_dir
    os.environ['CORENLP_HOME'] = stanfordnlp_dir
    aser_extractor = SeedRuleASERExtractorForConversation(
        corenlp_path=stanfordnlp_dir,
        corenlp_port=9000,
        relation_classifier_dir=relation_classifier_dir
    )
    threshold = 0.8

    data = load_json(args.data_dir)
    targets, outputs, rels = [], [], []
    contexts, utterances = [], []
    for sample in tqdm(data):
        context = sample['history']
        utt = sample['response']
        contexts.append(context)
        utterances.append(utt)
        for r in RELATIONS:
            targets.append(sample['tuples'][r])
            rels.append(r)

    extracted_tuples = aser_evt(contexts, utterances, threshold, stanfordnlp_dir, relation_classifier_dir)
    for tuples in extracted_tuples:
        tuples_by_rel = {r: [] for r in RELATIONS}
        for t in tuples:
            tuples_by_rel[t[1]].append(t)
        for r in RELATIONS:
            outputs.append(tuples_by_rel[r])

    aser_extractor.close()

    result = compute_metrics(targets, outputs, rels)
    print(result)
    df = {
        'context': contexts,
        'utterance': utterances,
        'extracted_tuples': extracted_tuples
    }
    df = pd.DataFrame(df)
    df.to_csv(args.saved_tuple_path, index=False)


if __name__ == '__main__':
    benchmark()
