from itertools import chain

import os
import torch
from copy import deepcopy
from scipy.special import softmax
from torchtext.data import get_tokenizer
from torchtext.vocab import GloVe

from eventualization.train_lstm_relation_classifier import LSTMRelationPredictor, RELATIONS
from .eventuality_extractor import SeedRuleEventualityExtractor
from .relation_extractor import SeedRuleRelationExtractor
from .utils import ANNOTATORS
from .utils import parse_sentense_with_stanford, get_corenlp_client


class BaseASERExtractor(object):
    """ Base ASER Extractor to extract both eventualities and relations.
    It includes an instance of `BaseEventualityExtractor` and an instance of `BaseRelationExtractor`.

    """

    def __init__(self, corenlp_path="", corenlp_port=0, **kw):
        """

        :param corenlp_path: corenlp path, e.g., /home/xliucr/stanford-corenlp-3.9.2
        :type corenlp_path: str (default = "")
        :param corenlp_port: corenlp port, e.g., 9000
        :type corenlp_port: int (default = 0)
        :param kw: other parameters
        :type kw: Dict[str, object]
        """

        self.corenlp_path = corenlp_path
        self.corenlp_port = corenlp_port
        self.annotators = kw.get("annotators", list(ANNOTATORS))

        _, self.is_externel_corenlp = get_corenlp_client(corenlp_path=self.corenlp_path, corenlp_port=self.corenlp_port)

        self.eventuality_extractor = None
        self.relation_extractor = None

    def close(self):
        """ Close the extractor safely
        """

        if not self.is_externel_corenlp:
            corenlp_client, _ = get_corenlp_client(corenlp_path=self.corenlp_path, corenlp_port=self.corenlp_port)
            corenlp_client.stop()
        if self.eventuality_extractor:
            self.eventuality_extractor.close()
        if self.relation_extractor:
            self.relation_extractor.close()

    def __del__(self):
        self.close()

    def parse_text(self, text, annotators=None):
        """ Parse a raw text by corenlp

        :param text: a raw text
        :type text: str
        :param annotators: annotators for corenlp, please refer to https://stanfordnlp.github.io/CoreNLP/annotators.html
        :type annotators: Union[List, None] (default = None)
        :return: the parsed result
        :rtype: List[Dict[str, object]]

        .. highlight:: python
        .. code-block:: python

            Input:

            "My army will find your boat. In the meantime, I'm sure we could find you suitable accommodations."

            Output:

            [{'dependencies': [(1, 'nmod:poss', 0),
                               (3, 'nsubj', 1),
                               (3, 'aux', 2),
                               (3, 'dobj', 5),
                               (3, 'punct', 6),
                               (5, 'nmod:poss', 4)],
              'lemmas': ['my', 'army', 'will', 'find', 'you', 'boat', '.'],
              'mentions': [],
              'ners': ['O', 'O', 'O', 'O', 'O', 'O', 'O'],
              'parse': '(ROOT (S (NP (PRP$ My) (NN army)) (VP (MD will) (VP (VB find) (NP '
                       '(PRP$ your) (NN boat)))) (. .)))',
              'pos_tags': ['PRP$', 'NN', 'MD', 'VB', 'PRP$', 'NN', '.'],
              'text': 'My army will find your boat.',
              'tokens': ['My', 'army', 'will', 'find', 'your', 'boat', '.']},
             {'dependencies': [(2, 'case', 0),
                               (2, 'det', 1),
                               (6, 'nmod:in', 2),
                               (6, 'punct', 3),
                               (6, 'nsubj', 4),
                               (6, 'cop', 5),
                               (6, 'ccomp', 9),
                               (6, 'punct', 13),
                               (9, 'nsubj', 7),
                               (9, 'aux', 8),
                               (9, 'iobj', 10),
                               (9, 'dobj', 12),
                               (12, 'amod', 11)],
              'lemmas': ['in',
                         'the',
                         'meantime',
                         ',',
                         'I',
                         'be',
                         'sure',
                         'we',
                         'could',
                         'find',
                         'you',
                         'suitable',
                         'accommodation',
                         '.'],
              'mentions': [],
              'ners': ['O',
                       'O',
                       'O',
                       'O',
                       'O',
                       'O',
                       'O',
                       'O',
                       'O',
                       'O',
                       'O',
                       'O',
                       'O',
                       'O'],
              'parse': '(ROOT (S (PP (IN In) (NP (DT the) (NN meantime))) (, ,) (NP (PRP '
                       "I)) (VP (VBP 'm) (ADJP (JJ sure) (SBAR (S (NP (PRP we)) (VP (MD "
                       'could) (VP (VB find) (NP (PRP you)) (NP (JJ suitable) (NNS '
                       'accommodations)))))))) (. .)))',
              'pos_tags': ['IN',
                           'DT',
                           'NN',
                           ',',
                           'PRP',
                           'VBP',
                           'JJ',
                           'PRP',
                           'MD',
                           'VB',
                           'PRP',
                           'JJ',
                           'NNS',
                           '.'],
              'text': "In the meantime, I'm sure we could find you suitable "
                      'accommodations.',
              'tokens': ['In',
                         'the',
                         'meantime',
                         ',',
                         'I',
                         "'m",
                         'sure',
                         'we',
                         'could',
                         'find',
                         'you',
                         'suitable',
                         'accommodations',
                         '.']}]
        """
        if annotators is None:
            annotators = self.annotators

        corenlp_client, _ = get_corenlp_client(
            corenlp_path=self.corenlp_path, corenlp_port=self.corenlp_port, annotators=annotators
        )
        parsed_result = parse_sentense_with_stanford(text, corenlp_client, self.annotators)
        return parsed_result

    def extract_eventualities_from_parsed_result(self, parsed_result, output_format="Eventuality", in_order=True, **kw):
        """ Extract eventualities from the parsed result

        :param parsed_result: the parsed result returned by corenlp
        :type parsed_result: List[Dict[str, object]]
        :param output_format: which format to return, "Eventuality" or "json"
        :type output_format: str (default = "Eventuality")
        :param in_order: whether the returned order follows the input token order
        :type in_order: bool (default = True)
        :param kw: other parameters
        :type kw: Dict[str, object]
        :return: the extracted eventualities
        :rtype: Union[List[List[aser.eventuality.Eventuality]], List[List[Dict[str, object]]], List[aser.eventuality.Eventuality], List[Dict[str, object]]]

        .. highlight:: python
        .. code-block:: python

            Input:

            [{'dependencies': [(1, 'nmod:poss', 0),
                               (3, 'nsubj', 1),
                               (3, 'aux', 2),
                               (3, 'dobj', 5),
                               (3, 'punct', 6),
                               (5, 'nmod:poss', 4)],
              'lemmas': ['my', 'army', 'will', 'find', 'you', 'boat', '.'],
              'mentions': [],
              'ners': ['O', 'O', 'O', 'O', 'O', 'O', 'O'],
              'parse': '(ROOT (S (NP (PRP$ My) (NN army)) (VP (MD will) (VP (VB find) (NP '
                       '(PRP$ your) (NN boat)))) (. .)))',
              'pos_tags': ['PRP$', 'NN', 'MD', 'VB', 'PRP$', 'NN', '.'],
              'text': 'My army will find your boat.',
              'tokens': ['My', 'army', 'will', 'find', 'your', 'boat', '.']},
             {'dependencies': [(2, 'case', 0),
                               (2, 'det', 1),
                               (6, 'nmod:in', 2),
                               (6, 'punct', 3),
                               (6, 'nsubj', 4),
                               (6, 'cop', 5),
                               (6, 'ccomp', 9),
                               (6, 'punct', 13),
                               (9, 'nsubj', 7),
                               (9, 'aux', 8),
                               (9, 'iobj', 10),
                               (9, 'dobj', 12),
                               (12, 'amod', 11)],
              'lemmas': ['in',
                         'the',
                         'meantime',
                         ',',
                         'I',
                         'be',
                         'sure',
                         'we',
                         'could',
                         'find',
                         'you',
                         'suitable',
                         'accommodation',
                         '.'],
              'mentions': [],
              'ners': ['O',
                       'O',
                       'O',
                       'O',
                       'O',
                       'O',
                       'O',
                       'O',
                       'O',
                       'O',
                       'O',
                       'O',
                       'O',
                       'O'],
              'parse': '(ROOT (S (PP (IN In) (NP (DT the) (NN meantime))) (, ,) (NP (PRP '
                       "I)) (VP (VBP 'm) (ADJP (JJ sure) (SBAR (S (NP (PRP we)) (VP (MD "
                       'could) (VP (VB find) (NP (PRP you)) (NP (JJ suitable) (NNS '
                       'accommodations)))))))) (. .)))',
              'pos_tags': ['IN',
                           'DT',
                           'NN',
                           ',',
                           'PRP',
                           'VBP',
                           'JJ',
                           'PRP',
                           'MD',
                           'VB',
                           'PRP',
                           'JJ',
                           'NNS',
                           '.'],
              'text': "In the meantime, I'm sure we could find you suitable "
                      'accommodations.',
              'tokens': ['In',
                         'the',
                         'meantime',
                         ',',
                         'I',
                         "'m",
                         'sure',
                         'we',
                         'could',
                         'find',
                         'you',
                         'suitable',
                         'accommodations',
                         '.']}]

            Output:

            [[my army will find you boat],
             [i be sure, we could find you suitable accommodation]]

        """

        if output_format not in ["Eventuality", "json"]:
            raise ValueError(
                "Error: extract_eventualities_from_parsed_result only supports Eventuality or json."
            )

        return self.eventuality_extractor.extract_from_parsed_result(
            parsed_result, output_format=output_format, in_order=in_order, **kw
        )

    def extract_eventualities_from_text(self, text, output_format="Eventuality", in_order=True, annotators=None, **kw):
        """ Extract eventualities from a raw text

        :param text: a raw text
        :type text: str
        :param output_format: which format to return, "Eventuality" or "json"
        :type output_format: str (default = "Eventuality")
        :param in_order: whether the returned order follows the input token order
        :type in_order: bool (default = True)
        :param annotators: annotators for corenlp, please refer to https://stanfordnlp.github.io/CoreNLP/annotators.html
        :type annotators: Union[List, None] (default = None)
        :param kw: other parameters
        :type kw: Dict[str, object]
        :return: the extracted eventualities
        :rtype: Union[List[List[aser.eventuality.Eventuality]], List[List[Dict[str, object]]], List[aser.eventuality.Eventuality], List[Dict[str, object]]]

        .. highlight:: python
        .. code-block:: python

            Input:

            "My army will find your boat. In the meantime, I'm sure we could find you suitable accommodations."

            Output:

            [[my army will find you boat],
             [i be sure, we could find you suitable accommodation]]
        """

        if output_format not in ["Eventuality", "json"]:
            raise NotImplementedError("Error: extract_eventualities_from_text only supports Eventuality or json.")

        parsed_result = self.parse_text(text, annotators=annotators)
        return self.extract_eventualities_from_parsed_result(
            parsed_result, output_format=output_format, in_order=in_order, **kw
        )

    def extract_relations_from_parsed_result(
            self, parsed_result, para_eventualities, output_format="Relation", in_order=True, **kw
    ):
        """ Extract relations from a parsed result (of a paragraph) and extracted eventualities

        :param parsed_result: the parsed result returned by corenlp
        :type parsed_result: List[Dict[str, object]]
        :param para_eventualities: eventualities in the paragraph
        :type para_eventualities: List[aser.eventuality.Eventuality]
        :param output_format: which format to return, "Relation" or "triplet"
        :type output_format: str (default = "Relation")
        :param in_order: whether the returned order follows the input token order
        :type in_order: bool (default = True)
        :param kw: other parameters
        :type kw: Dict[str, object]
        :return: the extracted relations
        :rtype: Union[List[List[aser.relation.Relation]], List[List[Dict[str, object]]], List[aser.relation.Relation], List[Dict[str, object]]]

        .. highlight:: python
        .. code-block:: python

            Input:

                [{'dependencies': [(1, 'nmod:poss', 0),
                                   (3, 'nsubj', 1),
                                   (3, 'aux', 2),
                                   (3, 'dobj', 5),
                                   (3, 'punct', 6),
                                   (5, 'nmod:poss', 4)],
                  'lemmas': ['my', 'army', 'will', 'find', 'you', 'boat', '.'],
                  'mentions': [],
                  'ners': ['O', 'O', 'O', 'O', 'O', 'O', 'O'],
                  'parse': '(ROOT (S (NP (PRP$ My) (NN army)) (VP (MD will) (VP (VB find) (NP '
                           '(PRP$ your) (NN boat)))) (. .)))',
                  'pos_tags': ['PRP$', 'NN', 'MD', 'VB', 'PRP$', 'NN', '.'],
                  'text': 'My army will find your boat.',
                  'tokens': ['My', 'army', 'will', 'find', 'your', 'boat', '.']},
                 {'dependencies': [(2, 'case', 0),
                                   (2, 'det', 1),
                                   (6, 'nmod:in', 2),
                                   (6, 'punct', 3),
                                   (6, 'nsubj', 4),
                                   (6, 'cop', 5),
                                   (6, 'ccomp', 9),
                                   (6, 'punct', 13),
                                   (9, 'nsubj', 7),
                                   (9, 'aux', 8),
                                   (9, 'iobj', 10),
                                   (9, 'dobj', 12),
                                   (12, 'amod', 11)],
                  'lemmas': ['in',
                             'the',
                             'meantime',
                             ',',
                             'I',
                             'be',
                             'sure',
                             'we',
                             'could',
                             'find',
                             'you',
                             'suitable',
                             'accommodation',
                             '.'],
                  'mentions': [],
                  'ners': ['O',
                           'O',
                           'O',
                           'O',
                           'O',
                           'O',
                           'O',
                           'O',
                           'O',
                           'O',
                           'O',
                           'O',
                           'O',
                           'O'],
                  'parse': '(ROOT (S (PP (IN In) (NP (DT the) (NN meantime))) (, ,) (NP (PRP '
                           "I)) (VP (VBP 'm) (ADJP (JJ sure) (SBAR (S (NP (PRP we)) (VP (MD "
                           'could) (VP (VB find) (NP (PRP you)) (NP (JJ suitable) (NNS '
                           'accommodations)))))))) (. .)))',
                  'pos_tags': ['IN',
                               'DT',
                               'NN',
                               ',',
                               'PRP',
                               'VBP',
                               'JJ',
                               'PRP',
                               'MD',
                               'VB',
                               'PRP',
                               'JJ',
                               'NNS',
                               '.'],
                  'text': "In the meantime, I'm sure we could find you suitable "
                          'accommodations.',
                  'tokens': ['In',
                             'the',
                             'meantime',
                             ',',
                             'I',
                             "'m",
                             'sure',
                             'we',
                             'could',
                             'find',
                             'you',
                             'suitable',
                             'accommodations',
                             '.']}],
                [[my army will find you boat],
                 [i be sure, we could find you suitable accommodation]]

                Output:

                [[],
                 [(7d9ea9023b66a0ebc167f0dbb6ea8cd75d7b46f9, 25edad6781577dcb3ba715c8230416fb0d4c45c4, {'Co_Occurrence': 1.0})],
                 [(8540897b645962964fd644242d4cc0032f024e86, 25edad6781577dcb3ba715c8230416fb0d4c45c4, {'Synchronous': 1.0})]]
        """

        if output_format not in ["Relation", "triplet"]:
            raise NotImplementedError("Error: extract_relations_from_parsed_result only supports Relation or triplet.")

        return self.relation_extractor.extract_from_parsed_result(
            parsed_result, para_eventualities, output_format=output_format, in_order=in_order, **kw
        )

    def extract_relations_from_text(self, text, output_format="Relation", in_order=True, annotators=None, **kw):
        """ Extract relations from a raw text and extracted eventualities

        :param text: a raw text
        :type text: str
        :param output_format: which format to return, "Relation" or "triplet"
        :type output_format: str (default = "Relation")
        :param in_order: whether the returned order follows the input token order
        :type in_order: bool (default = True)
        :param annotators: annotators for corenlp, please refer to https://stanfordnlp.github.io/CoreNLP/annotators.html
        :type annotators: Union[List, None] (default = None)
        :param kw: other parameters
        :type kw: Dict[str, object]
        :return: the extracted relations
        :rtype: Union[List[List[aser.relation.Relation]], List[List[Dict[str, object]]], List[aser.relation.Relation], List[Dict[str, object]]]

        .. highlight:: python
        .. code-block:: python

            Input:

            "My army will find your boat. In the meantime, I'm sure we could find you suitable accommodations."

            Output:

            [[],
             [(7d9ea9023b66a0ebc167f0dbb6ea8cd75d7b46f9, 25edad6781577dcb3ba715c8230416fb0d4c45c4, {'Co_Occurrence': 1.0})],
             [(8540897b645962964fd644242d4cc0032f024e86, 25edad6781577dcb3ba715c8230416fb0d4c45c4, {'Synchronous': 1.0})]]
        """

        if output_format not in ["Relation", "triplet"]:
            raise NotImplementedError("Error: extract_relations_from_text only supports Relation or triplet.")

        parsed_result = self.parse_text(text, annotators=annotators)
        para_eventualities = self.extract_eventualities_from_parsed_result(parsed_result)
        return self.extract_relations_from_parsed_result(
            parsed_result, para_eventualities, output_format=output_format, in_order=in_order, **kw
        )

    def extract_from_parsed_result(
            self,
            parsed_result,
            eventuality_output_format="Eventuality",
            relation_output_format="Relation",
            in_order=True,
            **kw
    ):
        """ Extract both eventualities and relations from a parsed result

        :param parsed_result: the parsed result returned by corenlp
        :type parsed_result: List[Dict[str, object]]
        :param eventuality_output_format: which format to return eventualities, "Eventuality" or "json"
        :type eventuality_output_format: str (default = "Eventuality")
        :param relation_output_format: which format to return relations, "Relation" or "triplet"
        :type relation_output_format: str (default = "Relation")
        :param in_order: whether the returned order follows the input token order
        :type in_order: bool (default = True)
        :param kw: other parameters
        :type kw: Dict[str, object]
        :return: the extracted eventualities and relations
        :rtype: Tuple[Union[List[List[aser.eventuality.Eventuality]], List[List[Dict[str, object]]], List[aser.eventuality.Eventuality], List[Dict[str, object]]], Union[List[List[aser.relation.Relation]], List[List[Dict[str, object]]], List[aser.relation.Relation], List[Dict[str, object]]]]

        .. highlight:: python
        .. code-block:: python

            Input:

            [{'dependencies': [(1, 'nmod:poss', 0),
                               (3, 'nsubj', 1),
                               (3, 'aux', 2),
                               (3, 'dobj', 5),
                               (3, 'punct', 6),
                               (5, 'nmod:poss', 4)],
              'lemmas': ['my', 'army', 'will', 'find', 'you', 'boat', '.'],
              'mentions': [],
              'ners': ['O', 'O', 'O', 'O', 'O', 'O', 'O'],
              'parse': '(ROOT (S (NP (PRP$ My) (NN army)) (VP (MD will) (VP (VB find) (NP '
                       '(PRP$ your) (NN boat)))) (. .)))',
              'pos_tags': ['PRP$', 'NN', 'MD', 'VB', 'PRP$', 'NN', '.'],
              'text': 'My army will find your boat.',
              'tokens': ['My', 'army', 'will', 'find', 'your', 'boat', '.']},
             {'dependencies': [(2, 'case', 0),
                               (2, 'det', 1),
                               (6, 'nmod:in', 2),
                               (6, 'punct', 3),
                               (6, 'nsubj', 4),
                               (6, 'cop', 5),
                               (6, 'ccomp', 9),
                               (6, 'punct', 13),
                               (9, 'nsubj', 7),
                               (9, 'aux', 8),
                               (9, 'iobj', 10),
                               (9, 'dobj', 12),
                               (12, 'amod', 11)],
              'lemmas': ['in',
                         'the',
                         'meantime',
                         ',',
                         'I',
                         'be',
                         'sure',
                         'we',
                         'could',
                         'find',
                         'you',
                         'suitable',
                         'accommodation',
                         '.'],
              'mentions': [],
              'ners': ['O',
                       'O',
                       'O',
                       'O',
                       'O',
                       'O',
                       'O',
                       'O',
                       'O',
                       'O',
                       'O',
                       'O',
                       'O',
                       'O'],
              'parse': '(ROOT (S (PP (IN In) (NP (DT the) (NN meantime))) (, ,) (NP (PRP '
                       "I)) (VP (VBP 'm) (ADJP (JJ sure) (SBAR (S (NP (PRP we)) (VP (MD "
                       'could) (VP (VB find) (NP (PRP you)) (NP (JJ suitable) (NNS '
                       'accommodations)))))))) (. .)))',
              'pos_tags': ['IN',
                           'DT',
                           'NN',
                           ',',
                           'PRP',
                           'VBP',
                           'JJ',
                           'PRP',
                           'MD',
                           'VB',
                           'PRP',
                           'JJ',
                           'NNS',
                           '.'],
              'text': "In the meantime, I'm sure we could find you suitable "
                      'accommodations.',
              'tokens': ['In',
                         'the',
                         'meantime',
                         ',',
                         'I',
                         "'m",
                         'sure',
                         'we',
                         'could',
                         'find',
                         'you',
                         'suitable',
                         'accommodations',
                         '.']}],
            [[my army will find you boat],
             [i be sure, we could find you suitable accommodation]]

            Output:

            ([[my army will find you boat],
              [i be sure, we could find you suitable accommodation]],
             [[],
              [(7d9ea9023b66a0ebc167f0dbb6ea8cd75d7b46f9, 25edad6781577dcb3ba715c8230416fb0d4c45c4, {'Co_Occurrence': 1.0})],
              [(8540897b645962964fd644242d4cc0032f024e86, 25edad6781577dcb3ba715c8230416fb0d4c45c4, {'Synchronous': 1.0})]])
        """

        if eventuality_output_format not in ["Eventuality", "json"]:
            raise NotImplementedError("Error: extract_eventualities only supports Eventuality or json.")
        if relation_output_format not in ["Relation", "triplet"]:
            raise NotImplementedError("Error: extract_relations only supports Relation or triplet.")

        if not isinstance(parsed_result, (list, tuple, dict)):
            raise NotImplementedError
        if isinstance(parsed_result, dict):
            is_single_sent = True
            parsed_result = [parsed_result]
        else:
            is_single_sent = False

        para_eventualities = self.extract_eventualities_from_parsed_result(
            parsed_result, output_format="Eventuality", in_order=True, **kw
        )
        para_relations = self.extract_relations_from_parsed_result(
            parsed_result, para_eventualities, output_format="Relation", in_order=True, **kw
        )

        if in_order:
            if eventuality_output_format == "json":
                para_eventualities = [[eventuality.encode(encoding=None) for eventuality in sent_eventualities] \
                                      for sent_eventualities in para_eventualities]
            if relation_output_format == "triplet":
                para_relations = [list(chain.from_iterable([relation.to_triplet() for relation in sent_relations])) \
                                  for sent_relations in para_relations]
            if is_single_sent:
                return para_eventualities[0], para_relations[0]
            else:
                return para_eventualities, para_relations
        else:
            eid2eventuality = dict()
            for eventuality in chain.from_iterable(para_eventualities):
                eid = eventuality.eid
                if eid not in eid2eventuality:
                    eid2eventuality[eid] = deepcopy(eventuality)
                else:
                    eid2eventuality[eid].update(eventuality)
            if eventuality_output_format == "Eventuality":
                eventualities = sorted(eid2eventuality.values(), key=lambda e: e.eid)
            elif eventuality_output_format == "json":
                eventualities = sorted(
                    [eventuality.encode(encoding=None) for eventuality in eid2eventuality.values()],
                    key=lambda e: e["eid"]
                )

            rid2relation = dict()
            for relation in chain.from_iterable(para_relations):
                if relation.rid not in rid2relation:
                    rid2relation[relation.rid] = deepcopy(relation)
                else:
                    rid2relation[relation.rid].update(relation)
            if relation_output_format == "Relation":
                para_relations = sorted(rid2relation.values(), key=lambda r: r.rid)
            elif relation_output_format == "triplet":
                para_relations = sorted(
                    chain.from_iterable([relation.to_triplets() for relation in rid2relation.values()]))
            return eventualities, para_relations

    def extract_from_text(
            self,
            text,
            eventuality_output_format="Eventuality",
            relation_output_format="Relation",
            in_order=True,
            annotators=None,
            **kw
    ):
        """ Extract both eventualities and relations from a raw text

        :param text: a raw text
        :type text: str
        :param eventuality_output_format: which format to return eventualities, "Eventuality" or "json"
        :type eventuality_output_format: str (default = "Eventuality")
        :param relation_output_format: which format to return relations, "Relation" or "triplet"
        :type relation_output_format: str (default = "Relation")
        :param in_order: whether the returned order follows the input token order
        :type in_order: bool (default = True)
        :param annotators: annotators for corenlp, please refer to https://stanfordnlp.github.io/CoreNLP/annotators.html
        :type annotators: Union[List, None] (default = None)
        :param kw: other parameters
        :type kw: Dict[str, object]
        :return: the extracted eventualities and relations
        :rtype: :rtype: Tuple[Union[List[List[aser.eventuality.Eventuality]], List[List[Dict[str, object]]], List[aser.eventuality.Eventuality], List[Dict[str, object]]], Union[List[List[aser.relation.Relation]], List[List[Dict[str, object]]], List[aser.relation.Relation], List[Dict[str, object]]]]

        .. highlight:: python
        .. code-block:: python

            Input:

            "My army will find your boat. In the meantime, I'm sure we could find you suitable accommodations."

            Output:

            ([[my army will find you boat],
              [i be sure, we could find you suitable accommodation]],
             [[],
              [(7d9ea9023b66a0ebc167f0dbb6ea8cd75d7b46f9, 25edad6781577dcb3ba715c8230416fb0d4c45c4, {'Co_Occurrence': 1.0})],
              [(8540897b645962964fd644242d4cc0032f024e86, 25edad6781577dcb3ba715c8230416fb0d4c45c4, {'Synchronous': 1.0})]])
        """
        if eventuality_output_format not in ["Eventuality", "json"]:
            raise NotImplementedError("Error: extract_eventualities only supports Eventuality or json.")
        if relation_output_format not in ["Relation", "triplet"]:
            raise NotImplementedError("Error: extract_relations only supports Relation or triplet.")

        parsed_result = self.parse_text(text, annotators=annotators)
        return self.extract_from_parsed_result(
            parsed_result,
            eventuality_output_format=eventuality_output_format,
            relation_output_format=relation_output_format,
            in_order=in_order,
            **kw
        )


class SeedRuleASERExtractor(BaseASERExtractor):
    """ ASER Extractor based on rules to extract both eventualities and relations (for ASER v1.0)

    """

    def __init__(self, corenlp_path="", corenlp_port=0, **kw):
        if "annotators" not in kw:
            kw["annotators"] = list(ANNOTATORS)
            if "parse" in kw["annotators"]:
                kw["annotators"].remove("parse")
            if "depparse" not in kw["annotators"]:
                kw["annotators"].append("depparse")
        super().__init__(corenlp_path, corenlp_port, **kw)
        from .rule import CLAUSE_WORDS
        self.eventuality_extractor = SeedRuleEventualityExtractor(
            corenlp_path=self.corenlp_path, corenlp_port=self.corenlp_port, skip_words=CLAUSE_WORDS, **kw
        )
        self.relation_extractor = SeedRuleRelationExtractor(**kw)


class SeedRuleASERExtractorForConversation(SeedRuleASERExtractor):
    """Wrapper of SeedRuleASERExtractor.

    Step 1: Use SeedRuleASERExtractor to extract eventualities from the response and the previous utterance.
    Step 2: Use a classifier trained with ATOMIC data to predict relation.
    """

    def __init__(self,
                 corenlp_path="",
                 corenlp_port=0,
                 relation_classifier_dir="models/relation_classifier_12relations",
                 max_sequence_length=32,
                 **kw):
        super().__init__(corenlp_path, corenlp_port, **kw)
        self.tokenizer = get_tokenizer('basic_english')
        self.glove_embed = GloVe(name='840B', dim=300)
        self.lstm_classifiers = {}
        for rel in RELATIONS:
            self.lstm_classifiers[rel] = LSTMRelationPredictor()
            self.lstm_classifiers[rel].load_state_dict(
                torch.load(os.path.join(relation_classifier_dir, f'{rel}.pth'), map_location='cpu'))

        if torch.cuda.is_available():
            self.lstm_classifiers = {k: v.cuda() for k, v in self.lstm_classifiers.items()}
            self.device = 'gpu'
        else:
            self.device = 'cpu'
        self.max_sequence_length = max_sequence_length
        self.relations = RELATIONS

    def predict_relation(self, head, tail, src):
        head_tok = self.tokenizer(head)
        head_tok = head_tok + [""] * (self.max_sequence_length - len(head_tok)) if len(
            head_tok) < self.max_sequence_length else head_tok[:self.max_sequence_length]
        tail_tok = self.tokenizer(tail)
        tail_tok = tail_tok + [""] * (self.max_sequence_length - len(tail_tok)) if len(
            tail_tok) < self.max_sequence_length else tail_tok[:self.max_sequence_length]
        src_tok = self.tokenizer(src)
        src_tok = src_tok + [""] * (self.max_sequence_length - len(src_tok)) if len(
            src_tok) < self.max_sequence_length else src_tok[:self.max_sequence_length]
        head_embed = self.glove_embed.get_vecs_by_tokens(head_tok).unsqueeze(0)
        tail_embed = self.glove_embed.get_vecs_by_tokens(tail_tok).unsqueeze(0)
        src_embed = self.glove_embed.get_vecs_by_tokens(src_tok).unsqueeze(0)
        if self.device == 'gpu':
            head_embed = head_embed.cuda()
            tail_embed = tail_embed.cuda()
            src_embed = src_embed.cuda()
        results = {}
        with torch.no_grad():
            for rel in RELATIONS:
                logits = self.lstm_classifiers[rel](head_embed, tail_embed, src_embed)
                logits = logits.cpu().numpy()
                score = softmax(logits, axis=1)[0]
                results[rel] = score[1]

        return results

    def extract_relations_from_conversation(self, prev_utt, utt, threshold, annotators=None, **kw):
        """
        Output a list of triples (head, relation, tail, is_single).
        """
        utt_results = self.extract_from_text(utt,
                                             in_order=False,
                                             annotators=annotators,
                                             **kw)
        prev_utt_results = self.extract_from_text(prev_utt,
                                                  in_order=False,
                                                  annotators=annotators,
                                                  **kw)
        utt_evt_cnt = len(utt_results[0])
        prev_utt_evt_cnt = len(prev_utt_results[0])

        valid_triplets = []
        # Loop around the events in the utterance to detect valid relation within the single utterance.
        for i in range(utt_evt_cnt):
            for j in range(i + 1, utt_evt_cnt):
                event1 = utt_results[0][i].__repr__()
                event2 = utt_results[0][j].__repr__()
                src = utt
                results = self.predict_relation(event1, event2, src)
                for k, v in results.items():
                    if v > threshold:
                        valid_triplets.append((event1, k, event2, True))
        # Loop around the events in the utterance and its previous utterance to detect valid relation within the
        # utterance pair.
        for i in range(utt_evt_cnt):
            for j in range(prev_utt_evt_cnt):
                event1 = utt_results[0][i].__repr__()
                event2 = prev_utt_results[0][j].__repr__()
                src = ' '.join([prev_utt, utt])
                results = self.predict_relation(event1, event2, src)
                for k, v in results.items():
                    if v > threshold:
                        valid_triplets.append((event1, k, event2, True))

        return valid_triplets
