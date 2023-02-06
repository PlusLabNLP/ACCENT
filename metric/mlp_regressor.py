"""Reference: https://aclanthology.org/2021.sigdial-1.13.pdf"""
import argparse
import pickle

import numpy as np
import pandas as pd
import spacy
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.neural_network import MLPRegressor
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelWithLMHead

from utils import load_json, dump_json, report_correlation

tagger = spacy.load('en_core_web_sm')
lemmatizer = WordNetLemmatizer()
STOPWORDS = stopwords.words('english')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_prepare_feature', action='store_true', help='Prepare features for the MLP regressor.')
    parser.add_argument('--raw_data_dir', nargs='+', help='Paths of the raw data.')
    parser.add_argument('--do_train', action='store_true', help='Train the MLP regressor on the training data.')
    parser.add_argument('--train_data_dir', type=str, help='Path of the training data.')
    parser.add_argument('--model_dir', type=str, help='Path of the MLP regressor.')
    parser.add_argument('--do_predict', action='store_true', help='Test the MLP regressor on the test data.')
    parser.add_argument('--test_data_dir', type=str, help='Path of the test data.')
    parser.add_argument('--debug', action='store_true', help='Save the predictions.')
    parser.add_argument('--saved_result_path', type=str, help='Path of the saved results.', default='mlp_regressor_results.csv')

    args = parser.parse_args()
    return args


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


def build_graph(concept_net_dir):
    """Represent the ConceptNet as
    {
        head: [tail1, tail2, ...],
        ...
    }
    Also, pre-calculate the two-hop relation graph for faster search.
    """
    df = pd.read_csv(concept_net_dir)
    one_hop_net = {}
    # Build one-hop relation graph.
    for i in tqdm(range(len(df))):
        if df.at[i, 'head'] not in one_hop_net:
            one_hop_net[df.at[i, 'head']] = []
        if df.at[i, 'tail'] not in one_hop_net[df.at[i, 'head']]:
            one_hop_net[df.at[i, 'head']].append(df.at[i, 'tail'])
    # Build two-hop relation graph.
    two_hop_net = {}
    for k in tqdm(one_hop_net.keys()):
        if k not in two_hop_net:
            two_hop_net[k] = []
        for tail in one_hop_net[k]:
            if tail in one_hop_net:
                two_hop_net[k].extend(one_hop_net[tail])  # Two-hop tuples.

    return one_hop_net, two_hop_net


def concept_net_search(concept_group1, concept_group2, one_hop_net, two_hop_net):
    one_hop_cnt = 0
    two_hop_cnt = 0
    for head in concept_group1:
        for tail in concept_group2:
            if head in one_hop_net and tail in one_hop_net[head]:
                one_hop_cnt += 1
            if head in two_hop_net and tail in two_hop_net[head]:
                two_hop_cnt += 1
    for head in concept_group2:
        for tail in concept_group1:
            if head in one_hop_net and tail in one_hop_net[head]:
                one_hop_cnt += 1
            if head in two_hop_net and tail in two_hop_net[head]:
                two_hop_cnt += 1

    return one_hop_cnt, two_hop_cnt


def dialogpt_scoring(context, response, model, tokenizer):
    contexts = []
    for c in context.split('</UTT>'):
        if c != '':
            contexts.append(c)
    contexts = ' </UTT> '.join(contexts)
    # Score the response.
    input_ids = tokenizer(response, return_tensors='pt').input_ids.to(model.device)
    response_score = model(input_ids, labels=input_ids)[0].item()
    # Score the concatenation of the context and the response.
    input_ids = tokenizer(contexts + ' </UTT> ' + response, return_tensors='pt').input_ids.to(model.device)
    concatenation_score = model(input_ids, labels=input_ids)[0].item()
    return response_score, concatenation_score


def prepare_features(src_file, tgt_file, one_hop_net, two_hop_net):
    """
    feature = [one-hop tuple number, two-hop tuple number, response length,
                DialoGPT response score, DialoGPT history + response score]
    """
    data = load_json(src_file)
    dialogpt_dir = "microsoft/DialoGPT-large"
    tokenizer = AutoTokenizer.from_pretrained(dialogpt_dir)
    model = AutoModelWithLMHead.from_pretrained(dialogpt_dir)
    for sample in tqdm(data):
        history = sample['history']
        response = sample['response']
        concept_group1 = []
        for s in history.split('</UTT>'):
            if s != '':
                tmp = extract_concept(s)
                concept_group1.extend(list(tmp))
        concept_group2 = extract_concept(response)
        one_hop_cnt, two_hop_cnt = concept_net_search(concept_group1, concept_group2, one_hop_net, two_hop_net)
        response_score, concatenation_score = dialogpt_scoring(history, response, model, tokenizer)
        sample['feature'] = [one_hop_cnt, two_hop_cnt, len(response.split()), response_score, concatenation_score]

    dump_json(data, tgt_file)


def mlp_regressor_train(X_train, y_train, model_dir):
    regressor = MLPRegressor(random_state=2022, max_iter=2000, early_stopping=True, validation_fraction=0.2).fit(
        X_train, y_train)
    pickle.dump(regressor, open(model_dir, 'wb'))


def mlp_regressor_test(X_test, y_test, model_dir):
    regressor = pickle.load(open(model_dir, 'rb'))
    predictions = regressor.predict(X_test)
    report_correlation(predictions, y_test)

    return predictions


def main():
    args = get_args()
    if args.do_prepare_feature:
        concept_net_dir = 'data/conceptnet_en_one_concept.csv'
        one_hop_net, two_hop_net = build_graph(concept_net_dir)
        for path in args.raw_data_dir:
            prepare_features(path, path.split('.')[0] + '_with_features.json', one_hop_net, two_hop_net)
    if args.do_train:
        # Load features.
        train_data = load_json(args.train_data_dir)
        X_train = []
        event_cs_train = []
        for sample in train_data:
            X_train.append(sample['feature'])
            event_cs_train.append(sample['event_cs'])
        X_train = np.array(X_train)
        event_cs_train = np.array(event_cs_train)
        mlp_regressor_train(X_train, event_cs_train, args.model_dir)
    if args.do_predict:
        # Load features.
        test_data = load_json(args.test_data_dir)
        X_test = []
        event_cs_test = []
        for sample in test_data:
            X_test.append(sample['feature'])
            event_cs_test.append(sample['event_cs'])
        X_test = np.array(X_test)
        event_cs_test = np.array(event_cs_test)
        auto_scores = mlp_regressor_test(X_test, event_cs_test, args.model_dir)
        if args.debug:
            if 'source' in test_data[0]:
                df = {
                    'history': [],
                    'response': [],
                    'source': [],
                    'system': [],
                    'gt_score': [],
                    'auto_score': [],
                }
                for i in range(len(test_data)):
                    df['history'].append(test_data[i]['history'])
                    df['response'].append(test_data[i]['response'])
                    df['source'].append(test_data[i]['source'])
                    df['system'].append(test_data[i]['system'])
                    df['gt_score'].append(test_data[i]['event_cs'])
                    df['auto_score'].append(auto_scores[i])
            else:
                df = {
                    'history': [],
                    'response': [],
                    'gt_score': [],
                    'auto_score': [],
                }
                for i in range(len(test_data)):
                    df['history'].append(test_data[i]['history'])
                    df['response'].append(test_data[i]['response'])
                    df['gt_score'].append(test_data[i]['event_cs'])
                    df['auto_score'].append(auto_scores[i])
            df = pd.DataFrame(df)
            df.to_csv(args.saved_result_path, index=False)


if __name__ == '__main__':
    main()
