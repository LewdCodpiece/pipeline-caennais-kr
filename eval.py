"""
Module d'évaluation de la diarisation sur le corpus CAENNAIS.
Deux types d'évaluation:
    - évaluation sur l'intégralité du corpus pour modèles out of the box
    - évaluation d'un modèle fine-tuneé sur le corpus CAENNAIS
Pour le deuxième type d'évaluation, on divise le corpus en 70% train, 15% de test/eval et 15% de dev
Le modèle est entrainé sur train, et est utilisé pour annoter test, c'est cette diarisation qui est évaluée en DER

Le data set est préparé par prep_dataset, ce script ne fait que la partie évaluation.
"""

import prep_dataset as prep

from os import listdir, makedirs
from pyannote.core import Annotation # pour des annotations lisibles par par pyannote.metrics
from pyannote.metrics.diarization import DiarizationErrorRate
from pympi.Elan import Eaf # pour les opérations sur les transcription eaf


def evaluer_corpus_entier(chemin_gold: str, chemin_hypothèse: str):
    annotations_gold: dict[str, Annotation] = {}
    for f in listdir(chemin_gold):
        if ".eaf" in f:
            annotations_gold[f[:5]] = prep.extraire_annotation_eaf(Eaf(chemin_gold+"/"+f), est_gold=True)
    
    annotations_hypothèses: dict[str, Annotation] = {}
    for f in listdir(chemin_hypothèse):
        if ".eaf" in f:
            annotations_hypothèses[f[:5]] = prep.extraire_annotation_eaf(Eaf(chemin_hypothèse+"/"+f))
    
    metric = DiarizationErrorRate()
    for nom, annotation in annotations_hypothèses.items():
        print(f"La DER de {nom} est de {metric(annotations_gold[nom], annotations_hypothèses[nom])}%.")


def evaluer(chemin_corpus: str, taille_train: float = .7, taille_test: float = .15, taille_dev: float = .15):
    prep.segmenter_corpus(chemin_corpus+"/audios", chemin_corpus+"/transcriptions", taille_train, taille_test, taille_dev)

    # on charche les annotations gold
    annotations_gold: dict[str, Annotation] = {}
    for f in listdir(chemin_corpus+"/test/gold"):
        if ".eaf" in f:
            annotations_gold[f[:5]] = prep.extraire_annotation_eaf(Eaf(chemin_corpus+"/test/gold/"+f), est_gold=True)
    
    # charge les annotations hypothèses
    annotations_hypo: dict[str, Annotation] = {}
    for f in listdir(chemin_corpus+"/test/annotation"):
        if ".eaf" in f:
            annotations_hypo[f[:5]] = prep.extraire_annotation_eaf(Eaf(chemin_corpus+"/test/transcriptions/"+f))
    
    metric = DiarizationErrorRate()
    for nom, annotation in annotations_hypo.items():
        print(f"La DER de {nom} est de {metric(annotations_gold[nom], annotations_hypo[nom])}%.")


def main():
    #evaluer_corpus_entier("./corpus/test/annotation/nemo/eaf", "./corpus/test/gold")

    evaluer_corpus_entier(
        "./corpus/dev/gold",
        "./corpus/dev/nemo/eaf"
    )


if __name__ == "__main__":
    main()