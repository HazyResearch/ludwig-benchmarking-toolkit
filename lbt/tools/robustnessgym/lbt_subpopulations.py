from lbt.tools.robustnessgym.base_subpopulation import BaseSubpopulation
from lbt.tools.robustnessgym import register_lbtsubpop
from robustnessgym import Spacy
from robustnessgym import ScoreSubpopulation, Identifier
import pandas as pd
from functools import partial

# TODO: ASN --> ADD Bias Pairs, Positive Words, Negative words, Identity Phrases, Emoji, NAUGHTY + OBSCENE PHRASES, NEgation


@register_lbtsubpop("entities")
class EntitySubpopulation(BaseSubpopulation):
    def __init__(self):
        self.name = "entities"
        self.entity_types = [
            "PERSON",
            "NORP",
            "FAC",
            "ORG",
            "GPE",
            "LOC",
            "PRODUCT",
            "EVENT",
            "WORK_OF_ART",
            "LAW",
            "LANGUAGE",
            "DATE",
            "TIME",
            "PERCENT",
            "MONEY",
            "QUANTITY",
            "ORDINAL",
            "CARDINAL",
        ]

    def score_fn(self, batch, columns, entity, spacy):
        try:
            entites_list = Spacy.retrieve(
                batch, columns, proc_fns=Spacy.entities
            )
        except ValueError:
            spacy_op = spacy(batch, columns)
            entites_list = Spacy.retrieve(
                spacy_op, columns, proc_fns=Spacy.entities
            )
        overall_batch_score = []
        for entities in entites_list:
            ents = set(entity["label"] for entity in entities)
            if entity in ents:
                overall_batch_score.append(1)
            else:
                overall_batch_score.append(0)
        return overall_batch_score

    def get_subpops(self, spacy):
        EntitiesSubpopulation = lambda entity, score_fn: ScoreSubpopulation(
            identifiers=[Identifier(f"{entity}")],
            intervals=[(1, 1)],
            score_fn=score_fn,
        )

        entity_subpops = []
        for entity in self.entity_types:
            entity_subpops.append(
                EntitiesSubpopulation(
                    entity, partial(self.score_fn, entity=entity, spacy=spacy)
                )
            )
        return entity_subpops


@register_lbtsubpop("pos")
class POSSubpopulation(BaseSubpopulation):
    def __init__(self):
        self.name = "POS"
        self.universalpos = [
            "ADJ",
            "ADP",
            "ADV",
            "AUX",
            "CONJ",
            "CCONJ",
            "DET",
            "INTJ",
            "NOUN",
            "NUM",
            "PART",
            "PRON",
            "PROPN",
            "PUNCT",
            "SCONJ",
            "SYM",
            "VERB",
            "X",
            "EOL",
            "SPACE",
        ]

    def score_fn(self, batch, columns, pos, spacy):
        try:
            spacy_annotations = Spacy.retrieve(batch, columns)
        except ValueError:
            spacy_op = spacy(batch, columns)
            spacy_annotations = Spacy.retrieve(spacy_op, columns)

        overall_batch_score = []
        for sample_annotation in spacy_annotations:
            pos_in_sample = set(
                token["pos"] for token in sample_annotation["tokens"]
            )
            if pos in pos_in_sample:
                overall_batch_score.append(1)
            else:
                overall_batch_score.append(0)

        return overall_batch_score

    def get_subpops(self, spacy):
        POSSubpopulation = lambda pos, score_fn: ScoreSubpopulation(
            identifiers=[Identifier(f"{pos}")],
            intervals=[(1, 1)],
            score_fn=score_fn,
        )

        pos_subpops = []
        for pos in self.universalpos:
            pos_subpops.append(
                POSSubpopulation(
                    pos, partial(self.score_fn, pos=pos, spacy=spacy)
                )
            )
        return pos_subpops