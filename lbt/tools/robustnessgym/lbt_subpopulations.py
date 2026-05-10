from lbt.tools.robustnessgym.base_subpopulation import BaseSubpopulation
from lbt.tools.robustnessgym import register_lbtsubpop
from robustnessgym import (
    LengthSubpopulation,
    HasPhrase,
    HasAnyPhrase,
)

import requests

from robustnessgym import Spacy
from robustnessgym import ScoreSubpopulation, Identifier
import pandas as pd
import itertools
from functools import partial

# TODO: ASN -->  Identity Phrases, Emoji,


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


@register_lbtsubpop("gender_bias")
class GenderBiasSubpopulation(BaseSubpopulation):
    def __init__(self):
        """
        Measures performance on gender co-occurence pairs
        """
        self.name = "gender_bias"
        self.female_identity = [
            "she",
            "her",
            "herself",
            "girl",
            "woman",
            "women",
            "females",
            "female",
            "girls",
            "feminine",
        ]
        self.male_identity = [
            "he",
            "him",
            "himself",
            "boy",
            "man",
            "men",
            "males",
            "male",
            "boys",
            "masculine",
        ]
        self.non_binary_identity = [
            "they",
            "them",
            "theirs",
            "their",
            "themself",
        ]
        self.gender_categories = {
            "female": self.female_identity,
            "male": self.male_identity,
            "non_binary": self.non_binary_identity,
        }

        self.career_words = [
            "executive",
            "professional",
            "corporation",
            "salary",
            "office",
            "business",
            "career",
        ]
        self.family_words = [
            "home",
            "parents",
            "children",
            "family",
            "cousin",
            "marriage",
            "wedding",
            "relatives",
        ]
        self.math_words = [
            "math",
            "algebra",
            "geometry",
            "calculus",
            "equation",
            "compute",
            "numbers",
            "addition",
        ]
        self.arts_words = [
            "poetry",
            "art",
            "dance",
            "literature",
            "novel",
            "symphony",
            "drama",
        ]
        self.science_words = [
            "science",
            "technology",
            "physics",
            "chemistry",
            "Einstein",
            "NASA",
            "experiment",
            "astronomy",
        ]

        self.domains = {
            "career": self.career_words,
            "family": self.family_words,
            "math": self.math_words,
            "arts": self.arts_words,
            "science": self.science_words,
        }

    def score_fn(self, batch, columns, pair):
        overall_batch_score = []
        for text in batch[columns[0]]:
            if pair[0] in text and pair[1] in text:
                overall_batch_score.append(1)
            else:
                overall_batch_score.append(0)
        return overall_batch_score

    def build_cooccurence_pairs(self, gender_categories: dict, domains: dict):
        bias_pairs = []
        for _, gender_list in gender_categories.items():
            for _, phrase_list in domains.items():
                bias_pairs.extend(
                    [
                        pair
                        for pair in itertools.product(gender_list, phrase_list)
                    ]
                )
        return bias_pairs

    def get_subpops(self, spacy):
        bias_pairs = self.build_cooccurence_pairs(
            self.gender_categories, self.domains
        )
        BiasCooccurenceSubpopulation = (
            lambda pair, score_fn: ScoreSubpopulation(
                identifiers=[Identifier(f"{pair[0]}_{pair[1]}")],
                intervals=[(1, 1)],
                score_fn=self.score_fn,
            )
        )

        bias_subpops = []
        for pair in bias_pairs:
            bias_subpops.append(
                BiasCooccurenceSubpopulation(
                    pair, partial(self.score_fn, pair=pair)
                )
            )
        return bias_subpops


@register_lbtsubpop("positive_sentiment")
class PositiveSentimentSubpopulation(BaseSubpopulation):
    def __init__(self):
        """
        Slice of dataset which contains positive sentiment carrying words
        """
        self.name = "positive_sentiment"
        self.positive_words_list = "https://gist.githubusercontent.com/mkulakowski2/4289437/raw/1bb4d7f9ee82150f339f09b5b1a0e6823d633958/positive-words.txt"

    def score_fn(self, batch, columns):
        pass

    def get_positive_words(self):
        response = requests.get(self.positive_words_list)
        _, words = (
            response.text.split("\n\n")[0],
            response.text.split("\n\n")[1],
        )
        word_list = words.split("\n")
        return word_list

    def get_subpops(self, spacy):
        return [
            HasAnyPhrase(
                phrase_groups=[self.get_positive_words()],
                identifiers=[Identifier("Positive Sentiment Words")],
            )
        ]


@register_lbtsubpop("negative_sentiment")
class NegativeSentimentSubpopulation(BaseSubpopulation):
    def __init__(self):
        """
        Slice of dataset which contains negative sentiment carrying words
        """
        self.name = "positive_sentiment"
        self.negative_words_list = "https://gist.githubusercontent.com/mkulakowski2/4289441/raw/dad8b64b307cd6df8068a379079becbb3f91101a/negative-words.txt"

    def score_fn(self, batch, columns):
        pass

    def get_negative_words(self):
        response = requests.get(self.negative_words_list)
        _, words = (
            response.text.split("\n\n")[0],
            response.text.split("\n\n")[1],
        )
        word_list = words.split("\n")
        return word_list

    def get_subpops(self, spacy):
        return [
            HasAnyPhrase(
                phrase_groups=[self.get_negative_words()],
                identifiers=[Identifier("Negative Sentiment Words")],
            )
        ]


@register_lbtsubpop("naughty_and_obscene")
class NaughtyObsceneSubpopulation(BaseSubpopulation):
    def __init__(self):
        """
        Slice of dataset which contains naught + obscene words
        """
        self.name = "naughty_and_obscene"
        self.word_list = "https://raw.githubusercontent.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/master/en"

    def score_fn(self, batch, columns):
        pass

    def get_naughty_obscene_word_list(self):
        response = requests.get(self.word_list)
        return response.text.split("\n")

    def get_subpops(self, spacy):
        return [
            HasAnyPhrase(
                phrase_groups=[self.get_naughty_obscene_word_list()],
                identifiers=[Identifier("Naughty and Obscene Words")],
            )
        ]


@register_lbtsubpop("sentence_length")
class SentenceLengthSubpopulation(BaseSubpopulation):
    def __init__(self):
        """
        Sentence length based slices
        """
        self.name = "sentence_length"

    def score_fn(self, batch, columns):
        pass

    def get_subpops(self, spacy):
        return [
            LengthSubpopulation(
                intervals=[
                    (0, 20),
                    (20, 40),
                    (40, 60),
                    (60, 80),
                    (80, 100),
                    (100, 120),
                    (120, 140),
                ]
            )
        ]
