"""Do one-hop searching in ATOMIC for eventualization."""
import argparse

import pandas as pd
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

from eventualization.evt_evaluate import RELATIONS, compute_metrics
from utils import load_json

tagger = spacy.load('en_core_web_sm')
lemmatizer = WordNetLemmatizer()

STOPWORDS = stopwords.words('english')


def prepare_atomic(atomic_dir):
    df = pd.read_csv(atomic_dir)
    df = df.dropna(axis=0, how='any')
    heads = []
    tails = []
    rels = []
    for i in tqdm(range(len(df))):
        if df.iloc[i]['rel'] in RELATIONS:
            heads.append(df.iloc[i]['head'])
            tails.append(df.iloc[i]['tail'])
            rels.append(df.iloc[i]['rel'])
    new_df = pd.DataFrame({
        'head': heads,
        'tail': tails,
        'rel': rels
    })

    print(f'ATOMIC subset contains {len(new_df)} tuples.')

    return new_df


def extract_concept(sentence):
    """Extract concepts from the input sentence.

    we use a part-of-speech (POS) tagger to find the nouns, verbs, and adjectives that are not stopwords
    and then construct a set of potential concepts by including the lemmatized version of these words."""
    results = tagger(sentence)
    concepts = []
    for word in results:
        if word.pos_ == 'VERB' or word.pos_ == 'NOUN' or word.pos_ == 'ADJ':
            w = word.text
            if w.lower() in STOPWORDS:  # Skip stop words.
                continue
            w = lemmatizer.lemmatize(w)  # Include the lemmatized version of the word.
            concepts.append(w.lower())
    concepts = set(concepts)
    return concepts


def search_atomic(atomic, context, utterance, with_single=True):
    """Do one-hop search in the ATOMIC.

    If with single is True, include the triplets where both the head and tail come from the current utterance."""
    prev_utt = context.split('</UTT>')
    prev_utt_concept = extract_concept(prev_utt[-2])
    current_utt_concept = extract_concept(utterance)
    candidate_triplets = []
    for c in current_utt_concept:
        for i in range(len(atomic)):
            flag = False
            if c in str(atomic.at[i, 'head']).lower():
                for cc in prev_utt_concept:
                    if cc == c:
                        continue
                    if cc in str(atomic.at[i, 'tail']).lower():
                        flag = True
                        break
                if not flag and with_single:
                    for cc in current_utt_concept:
                        if cc == c:
                            continue
                        if cc in str(atomic.at[i, 'tail']).lower():
                            flag = True
                            break
            elif c in str(atomic.at[i, 'tail']).lower():
                for cc in prev_utt_concept:
                    if cc == c:
                        continue
                    if cc in str(atomic.at[i, 'head']).lower():
                        flag = True
                        break
                if not flag and with_single:
                    for cc in current_utt_concept:
                        if cc == c:
                            continue
                        if cc in str(atomic.at[i, 'head']).lower():
                            flag = True
                            break
            if flag:
                candidate_triplets.append(
                    (atomic.at[i, 'head'], atomic.at[i, 'rel'], atomic.at[i, 'tail'])
                )

    return set(candidate_triplets)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_tuple_path', type=str, help='Event-relation extraction results dir',
                        default='one_hop_search_result.csv')
    parser.add_argument('--data_dir', type=str, help='Test data dir.',
                        default='data/deco/deco_test.json')
    parser.add_argument('--atomic_dir', type=str, help='ATOMIC dir.',
                        default='data/atomic2020_train_processed.csv')

    return parser.parse_args()


def main():
    args = get_args()
    atomic = prepare_atomic(args.atomic_dir)
    data = load_json(args.data_dir)
    targets, outputs, rels = [], [], []
    contexts, utterances, extracted_tuples = [], [], []
    for sample in tqdm(data):
        context = sample['history']
        utt = sample['response']
        contexts.append(context)
        utterances.append(utt)
        output = search_atomic(atomic, context, utt)
        extracted_tuples.append(output)
        tuples_by_rel = {r: [] for r in RELATIONS}
        for t in output:
            tuples_by_rel[t[1]].append(t)
        for r in RELATIONS:
            outputs.append(tuples_by_rel[r])
            targets.append(sample['tuples'][r])
            rels.append(r)
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
    main()
