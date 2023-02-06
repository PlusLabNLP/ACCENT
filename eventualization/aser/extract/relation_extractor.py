from itertools import chain

from copy import deepcopy

from .rule import SEED_CONNECTIVE_DICT
from ..relation import Relation, relation_senses


class BaseRelationExtractor(object):
    """ Base ASER relation rxtractor to extract relations

    """

    def __init__(self, **kw):
        pass

    def close(self):
        pass

    def __del__(self):
        self.close()

    def extract_from_parsed_result(
            self, parsed_result, para_eventualities, output_format="Relation", in_order=True, **kw
    ):
        """ Extract relations from the parsed result

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

        raise NotImplementedError


class SeedRuleRelationExtractor(BaseRelationExtractor):
    """ ASER relation extractor based on rules to extract relations (for ASER v1.0)

    """

    def __init__(self, **kw):
        super().__init__(**kw)

    def extract_from_parsed_result(
            self, parsed_result, para_eventualities, output_format="Relation", in_order=True, **kw
    ):
        if output_format not in ["Relation", "triplet"]:
            raise NotImplementedError("Error: extract_from_parsed_result only supports Relation or triplet.")

        connective_dict = kw.get("connective_dict", SEED_CONNECTIVE_DICT)

        para_relations = list()
        for sent_parsed_result, eventualities in zip(parsed_result, para_eventualities):
            relations_in_sent = list()
            for head_eventuality in eventualities:
                for tail_eventuality in eventualities:
                    if not head_eventuality.position < tail_eventuality.position:
                        continue
                    heid = head_eventuality.eid
                    teid = tail_eventuality.eid
                    extracted_senses = self._extract_from_eventuality_pair_in_one_sentence(
                        connective_dict, sent_parsed_result, head_eventuality, tail_eventuality
                    )
                    if len(extracted_senses) > 0:
                        relations_in_sent.append(Relation(heid, teid, extracted_senses))
            para_relations.append(relations_in_sent)

        for i in range(len(parsed_result) - 1):
            eventualities1, eventualities2 = para_eventualities[i], para_eventualities[i + 1]
            relations_between_sents = list()
            if len(eventualities1) == 1 and len(eventualities2) == 1:
                s1_tokens, s2_tokens = parsed_result[i]["tokens"], parsed_result[i + 1]["tokens"]
                s1_eventuality, s2_eventuality = eventualities1[0], eventualities2[0]
                heid, teid = s1_eventuality.eid, s2_eventuality.eid
                extracted_senses = self._extract_from_eventuality_pair_in_two_sentence(
                    connective_dict, s1_eventuality, s2_eventuality, s1_tokens, s2_tokens
                )
                if len(extracted_senses) > 0:
                    relations_between_sents.append(Relation(heid, teid, extracted_senses))
            para_relations.append(relations_between_sents)

        if in_order:
            if output_format == "triplet":
                para_relations = [sorted(chain.from_iterable([r.to_triplets() for r in relations]))
                                  for relations in para_relations]
            return para_relations
        else:
            if output_format == "Relation":
                rid2relation = dict()
                for relation in chain(*para_relations):
                    if relation.rid not in rid2relation:
                        rid2relation[relation.rid] = deepcopy(relation)
                    else:
                        rid2relation[relation.rid].update(relation)
                relations = sorted(rid2relation.values(), key=lambda r: r.rid)
            elif output_format == "triplet":
                relations = sorted([r.to_triplets() for relations in para_relations for r in relations])
            return relations

    def _extract_from_eventuality_pair_in_one_sentence(
            self, connective_dict, sent_parsed_result, head_eventuality, tail_eventuality
    ):
        extracted_senses = ['Co_Occurrence']
        for sense in relation_senses:
            for connective_words in connective_dict[sense]:
                if self._verify_connective_in_one_sentence(
                        connective_words, head_eventuality, tail_eventuality, sent_parsed_result["dependencies"],
                        sent_parsed_result["tokens"]
                ):
                    extracted_senses.append(sense)
                    break
        return extracted_senses

    def _extract_from_eventuality_pair_in_two_sentence(
            self, connective_dict, s1_eventuality, s2_eventuality, s1_tokens, s2_tokens
    ):
        extracted_senses = list()
        for sense in relation_senses:
            for connective_words in connective_dict[sense]:
                if self._verify_connective_in_two_sentence(
                        connective_words, s1_eventuality, s2_eventuality, s1_tokens, s2_tokens
                ):
                    extracted_senses.append(sense)
                    break

        return extracted_senses

    def _verify_connective_in_one_sentence(
            self, connective_words, head_eventuality, tail_eventuality, sentence_dependencies, sentence_tokens
    ):
        def get_connective_position(connective_words):
            tmp_positions = list()
            for w in connective_words:
                tmp_positions.append(sentence_tokens.index(w))
            return sum(tmp_positions) / len(tmp_positions) if tmp_positions else 0.0

        # Connective Words need to be presented in sentence
        if set(connective_words) - set(sentence_tokens):
            return False
        # Connective phrase need to be presented in sentence
        connective_string = " ".join(connective_words)
        sentence_string = " ".join(sentence_tokens)
        if connective_string not in sentence_string:
            return False
        shrinked_dependencies = self._shrink_sentence_dependencies(
            head_eventuality._raw_dependencies, tail_eventuality._raw_dependencies, sentence_dependencies
        )
        if not shrinked_dependencies:
            return False
        found_advcl = False
        for (governor, dep, dependent) in shrinked_dependencies:
            if governor == '_H_' and dependent == '_T_' and 'advcl' in dep:
                found_advcl = True
                break
        if not found_advcl:
            return False
        connective_position = get_connective_position(connective_words)
        e1_position, e2_position = head_eventuality.position, tail_eventuality.position
        if 'instead' not in connective_words:
            if e1_position < connective_position < e2_position:
                return True
            else:
                return False
        else:
            if e1_position < e2_position < connective_position:
                return True
            else:
                return False

    def _verify_connective_in_two_sentence(
            self, connective_words, s1_eventuality, s2_eventuality, s1_tokens, s2_tokens
    ):
        def get_connective_position():
            tmp_positions = list()
            for w in connective_words:
                if w in s1_tokens:
                    tmp_positions.append(s1_tokens.index(w))
                elif w in s2_tokens:
                    tmp_positions.append(s2_tokens.index(w) + len(s1_tokens))
            return sum(tmp_positions) / len(tmp_positions) if tmp_positions else 0.0

        sentence_tokens = s1_tokens + s2_tokens
        # Connective Words need to be presented in sentence
        if set(connective_words) - set(sentence_tokens):
            return False
        # Connective phrase need to be presented in sentence
        connective_string = " ".join(connective_words)
        sentence_string = " ".join(sentence_tokens)
        if connective_string not in sentence_string:
            return False
        connective_position = get_connective_position()
        e1_position, e2_position = s1_eventuality.position, \
                                   s2_eventuality.position + len(s1_tokens)
        if 'instead' not in connective_words:
            if e1_position < connective_position < e2_position and e2_position - e1_position < 10:
                return True
            else:
                return False
        else:
            if e1_position < e2_position < connective_position and e2_position - e1_position < 10:
                return True
            else:
                return False

    def _shrink_sentence_dependencies(self, head_dependencies, tail_dependencies, sentence_dependencies):
        head_nodes = set()
        for governor, _, dependent in head_dependencies:
            head_nodes.add(governor)
            head_nodes.add(dependent)
        tail_nodes = set()
        for governor, _, dependent in tail_dependencies:
            tail_nodes.add(governor)
            tail_nodes.add(dependent)
        if head_nodes & tail_nodes:
            return None

        new_dependencies = list()
        for governor, dep, dependent in sentence_dependencies:
            if governor in head_nodes:
                new_governor = '_H_'
            elif governor in tail_nodes:
                new_governor = '_T_'
            else:
                new_governor = governor
            if dependent in head_nodes:
                new_dependent = '_H_'
            elif dependent in tail_nodes:
                new_dependent = '_T_'
            else:
                new_dependent = dependent
            if new_governor != new_dependent:
                new_dependencies.append((new_governor, dep, new_dependent))
        return new_dependencies
