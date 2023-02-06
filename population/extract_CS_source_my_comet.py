import json

import torch
from scipy import spatial
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def dump_json(obj, file_name, encoding="utf-8", default=None):
    if default is None:
        with open(file_name, 'w', encoding=encoding) as fw:
            json.dump(obj, fw)
    else:
        with open(file_name, 'w', encoding=encoding) as fw:
            json.dump(obj, fw, default=default)


class CSKB():
    def __init__(self, comet_dir, threshold=0.92):
        print('inside initializer')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.relation_list = [
            'isAfter',
            'isBefore',
            'Causes',
            'xReact',
            'xAttr',
            'oWant',
            'xReason',
            'oReact',
            'xWant',
            'HasSubEvent',
            'HinderedBy',
            'gWant',
            'gEffect',
            'oEffect',
            'gReact',
            'xEffect',
            'xIntent',
            'xNeed'
        ]
        print('Loading SentBert model ...')
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2').to(self.device)
        print('Done!')
        print('Loading Comet-atomic2020-Bart model ...')
        self.comet_model = AutoModelForSeq2SeqLM.from_pretrained(comet_dir).to(self.device)
        self.comet_tokenizer = AutoTokenizer.from_pretrained(comet_dir)
        print('Done!')
        print('end of initializer')

    def extract_cs_knowledge(self, tuples):
        self.tuple_cs_document = {}
        progress_bar = tqdm(range(len(tuples)))
        for ind, (head, relation, tail) in enumerate(tuples):
            output_h_r_t = []
            if relation == 'general Want':
                relation = 'gWant'
            if relation == 'general Effect':
                relation = 'gEffect'
            if relation == 'general React':
                relation = 'gReact'

            assert relation in self.relation_list, f'Invalid relation {relation}!'

            output_tails = []
            gen_tails = self.comet_generate('{} {} [GEN]'.format(head, relation))
            output_tails += gen_tails

            for output_tail in output_tails:
                if (head, relation, output_tail) not in output_h_r_t:
                    output_h_r_t.append((head, relation, output_tail))
            self.tuple_cs_document[head + '\t' + relation + '\t' + tail] = output_h_r_t
            progress_bar.update(1)
        try:
            dump_json(self.tuple_cs_document, 'tmp.json')
        except Exception as e:
            import pdb
            pdb.set_trace()

    def trim_batch(self, input_ids, pad_token_id, attention_mask=None, ):
        """Remove columns that are populated exclusively by pad_token_id"""
        keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
        if attention_mask is None:
            return input_ids[:, keep_column_mask]
        else:
            return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])

    def comet_generate(self, query, decode_method="beam", num_beams=10, num_return_sequences=10):
        with torch.no_grad():
            query_token = self.comet_tokenizer(query, return_tensors="pt", truncation=True, padding="max_length").to(
                self.device)
            input_ids, attention_mask = self.trim_batch(**query_token, pad_token_id=self.comet_tokenizer.pad_token_id)

            gen_outputs = self.comet_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_start_token_id=None,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
            output = self.comet_tokenizer.batch_decode(gen_outputs, skip_special_tokens=True,
                                                       clean_up_tokenization_spaces=True)
            return output

    def compute_sensibility(self, tuples, gt_scores):
        """Currently consider the following scoring methods to compare tail and gen_tail."""
        print('computing sensibility scores')
        preds = {
            'cosine_sim': [],
        }

        progress_bar = tqdm(range(len(tuples)))
        for (head, relation, tail) in tuples:
            if relation == 'general Want':
                relation = 'gWant'
            if relation == 'general Effect':
                relation = 'gEffect'
            if relation == 'general React':
                relation = 'gReact'

            tail_proc = tail.lower()
            document = self.tuple_cs_document[head + '\t' + relation + '\t' + tail]
            cs_tail = [t[2].lower().strip() for t in document]
            max_similarity = 0
            if not cs_tail:
                print('the tuple can not be extracted from cs document therefore it is not sensible!!!!!!!!!!!!')
            else:
                # SBERT embeddings + cosine similarity.
                enc_tail = self.model.encode(tail_proc.lower().strip())
                enc_cs_tails = self.model.encode(cs_tail)
                for m, enc_cs_tail in enumerate(enc_cs_tails):
                    similarity = 1 - spatial.distance.cosine(enc_tail, enc_cs_tail)
                    if similarity > max_similarity:
                        max_similarity = similarity

            preds['cosine_sim'].append(max_similarity)
            progress_bar.update(1)

        return preds
