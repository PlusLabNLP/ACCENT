import argparse
import ast
import os

import pandas as pd
from tqdm import tqdm

from metric.accent import ACCENT
from utils import load_json, report_correlation


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evt_model_dir", type=str, help="Paths of the event-relation extractor.", required=True)
    parser.add_argument("--comet_dir", type=str, help="Paths of the Comet model.", required=True)
    parser.add_argument("--data_dir", type=str, help="Path of the json data.", required=True)
    parser.add_argument("--verbose", action='store_true', help='Verbose mode.')
    parser.add_argument("--cpu", action='store_true', help='Use CPU instead of GPU.')
    parser.add_argument("--use_saved_tuples", action='store_true', help='Start from the saved csv file.')
    parser.add_argument("--saved_tuple_path", type=str, help='Only necessary when use_saved_tuples is True',
                        default='for_debug_metric.csv')
    parser.add_argument("--num_beams", type=int, default=10, help="Beam size when querying the CSKB.")
    parser.add_argument("--embedder", help='The model to get the sentence embedding.',
                        choices=['sentence_bert', 'simcse'], default='sentence_bert')

    args = parser.parse_args()
    return args


def main(args):
    verbose_mode = args.verbose
    cs_scorer = ACCENT(
        comet_dir=args.comet_dir, evt_model_dir=args.evt_model_dir, use_gpu=(not args.cpu), embedder=args.embedder)
    data = load_json(args.data_dir)

    auto_scores = []
    gt_scores = []
    if verbose_mode:
        extracted_tuples = []
        cs_documents = []
        histories = []
        responses = []
    for sample in tqdm(data):
        if 'event_cs' in sample:
            gt_scores.append(sample['event_cs'])
        if verbose_mode:
            score, tuples, cs_doc, tuple_scores = cs_scorer.score(
                context=sample['history'],
                utterance=sample['response'],
                num_beams=args.num_beams,
                verbose_mode=verbose_mode
            )
            extracted_tuples.append(tuples)
            cs_documents.append(cs_doc)
            histories.append(sample['history'])
            responses.append(sample['response'])
        else:
            score = cs_scorer.score(
                context=sample['history'],
                utterance=sample['response'],
                num_beams=args.num_beams,
                verbose_mode=verbose_mode
            )
        auto_scores.append(score)

    if len(gt_scores) != 0:
        report_correlation(auto_scores, gt_scores)

    if verbose_mode:
        if len(gt_scores) != 0:
            verbose_results = pd.DataFrame({
                'history': histories,
                'response': responses,
                'gt_score': gt_scores,
                'auto_score': auto_scores,
                'extracted_tuples': extracted_tuples,
                'cs_documents': cs_documents
            })
        else:
            verbose_results = pd.DataFrame({
                'history': histories,
                'response': responses,
                'auto_score': auto_scores,
                'extracted_tuples': extracted_tuples,
                'cs_documents': cs_documents
            })
        os.makedirs('outputs', exist_ok=True)
        verbose_results.to_csv(os.path.join('outputs', args.saved_tuple_path), index=False)


def from_saved_file(args):
    """From saved extracted tuples and the cs documents.

    This is for the debugging purpose.
    """
    file_path = os.path.join('outputs', args.saved_tuple_path)
    comet_dir = args.comet_dir
    evt_model_dir = args.evt_model_dir
    cs_scorer = ACCENT(comet_dir=comet_dir, evt_model_dir=evt_model_dir, use_gpu=(not args.cpu), embedder=args.embedder)
    df = pd.read_csv(file_path)
    gt_scores = list(df['gt_score'])
    auto_scores = []
    for i in tqdm(range(len(df))):
        contexts = df.at[i, 'history'].split('</UTT>')[:-1]
        assert len(contexts) == 4
        extracted_tuples = ast.literal_eval(df.at[i, 'extracted_tuples'])
        cs_documents = ast.literal_eval(df.at[i, 'cs_documents'])
        assert len(extracted_tuples) == len(cs_documents)

        score = cs_scorer.score_with_symbolic_intermediate(extracted_tuples, cs_documents)

        auto_scores.append(score)

    report_correlation(auto_scores, gt_scores)


if __name__ == '__main__':
    args = get_args()
    if args.use_saved_tuples:
        from_saved_file(args)
    else:
        main(args)
