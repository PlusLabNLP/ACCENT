import torch
from scipy import spatial
from sentence_transformers import SentenceTransformer
from simcse import SimCSE
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from eventualization.few_shot_evt import Eventualization


class ACCENT:
    """An event-commonsense evaluation metric for open-domain dialogue systems."""

    def __init__(self, comet_dir, evt_model_dir, use_gpu=True, embedder='sentence_bert'):
        self.comet = AutoModelForSeq2SeqLM.from_pretrained(comet_dir)
        if use_gpu:
            self.comet = self.comet.cuda()
        self.comet_tokenizer = AutoTokenizer.from_pretrained(comet_dir)
        if embedder == 'sentence_bert':
            self.encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        elif embedder == 'simcse':
            self.encoder = SimCSE("princeton-nlp/sup-simcse-roberta-base")
        else:
            raise NotImplementedError(f'Embedder {embedder} is not supported!')

        self.evt = Eventualization(with_context=True, one_previous_utterance_only=True)
        self.evt_model = AutoModelForSeq2SeqLM.from_pretrained(evt_model_dir)
        if use_gpu:
            self.evt_model = self.evt_model.cuda()
        self.evt_tokenizer = AutoTokenizer.from_pretrained(evt_model_dir)

    def score(self, context, utterance, num_beams=10, max_length=64,
              verbose_mode=False, log_to_console=False):
        """Score the commonsense plausibility with the given dialogue history and the response.

        If verbose_mode is True, return final_scores, extracted_tuples, cs_documents;
        otherwise, return final_scores only.
        """
        tuples = self.get_symbolic_representation(context, utterance, max_length)

        if log_to_console:
            print('Eventualization results:')
            print(tuples)

        if verbose_mode:
            final_score, cs_documents, tuple_scores = self.score_with_given_tuples(
                tuples,
                num_beams=num_beams,
                max_length=max_length,
                verbose_mode=verbose_mode,
                log_to_console=log_to_console
            )

            return final_score, tuples, cs_documents, tuple_scores
        else:
            final_score = self.score_with_given_tuples(
                tuples,
                num_beams=num_beams,
                max_length=max_length,
                verbose_mode=verbose_mode,
                log_to_console=log_to_console
            )
            return final_score

    def get_symbolic_representation(self, context, utterance, max_length=64):
        """Extract the event-relation tuples from the target response and its dialogue history."""
        evt_results = self.evt.inference(
            sample=(context, utterance),
            model=self.evt_model,
            tokenizer=self.evt_tokenizer,
            max_length=max_length
        )

        # Post-process the generated outputs.
        tuples = []
        for relation, output in evt_results.items():
            if output == 'None':
                continue
            if 'event1:' not in output or 'event2:' not in output:
                # Invalid output.
                continue
            head = output[:output.find('event2:')]
            if head[-1] == ' ':
                head = head[:-1]
            if head[-1] == ';':
                head = head[:-1]
            tail = output[output.find('event2:'):]
            # Remove the prompts (event1: , event2: )
            head = head[8:]
            tail = tail[8:]
            if len(head) and len(tail):
                tuples.append((head, relation, tail))

        return tuples

    def score_with_symbolic_intermediate(self, tuples, cs_documents, verbose_mode=False):
        """Calculate the final score with two symbolic intermediates."""
        # Compute similarity.
        sim_scores = []
        for i in range(len(tuples)):
            max_similarity = 0

            cs_docs = cs_documents[i]
            if not cs_docs:
                print('the tuple can not be extracted from cs document therefore it is not sensible!')
            else:
                tail = tuples[i][2]
                encoded_cs_doc = self.encoder.encode(cs_docs)
                encoded_extracted_s = self.encoder.encode(tail)
                for m, encoded_cs_s in enumerate(encoded_cs_doc):
                    sim = 1 - spatial.distance.cosine(encoded_extracted_s, encoded_cs_s)
                    if sim > max_similarity:
                        max_similarity = sim
            sim_scores.append(max_similarity)

        # Aggregate the scores.
        cosine_sim_avg = sum(sim_scores) / len(sim_scores) if len(
            sim_scores) else 0.5  # If there is no extracted tuple, we set the score as 0.5.

        if verbose_mode:
            return cosine_sim_avg, sim_scores
        else:
            return cosine_sim_avg

    def score_with_given_tuples(self, tuples, num_beams=10, max_length=64,
                                verbose_mode=False, log_to_console=False):
        """
        tuples: [list of tuples with format (head, relation, tail)]

        If verbose_mode is true, return final_scores, cs_document; otherwise, return final_scores only.
        """
        # Get the reference tuples from COMET.
        cs_documents = []
        for x in tuples:
            s = f'{x[0]} {x[1]} [GEN]'
            inputs = self.comet_tokenizer(s, return_tensors='pt').to(self.comet.device)
            with torch.no_grad():
                outputs = self.comet.generate(
                    **inputs,
                    max_length=max_length,
                    early_stopping=True,
                    num_beams=num_beams,
                    num_return_sequences=num_beams,
                    no_repeat_ngram_size=2
                )
            decoded_outputs = []
            for output in outputs:
                sent = self.comet_tokenizer.decode(
                    output,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                decoded_outputs.append(sent)

            cs_documents.append(decoded_outputs)

        if log_to_console:
            print('Commonsense document:')
            print(cs_documents)

        if verbose_mode:
            final_score, tuple_scores = self.score_with_symbolic_intermediate(tuples, cs_documents, verbose_mode)
            return final_score, cs_documents, tuple_scores
        else:
            final_score = self.score_with_symbolic_intermediate(tuples, cs_documents, verbose_mode)
            return final_score
