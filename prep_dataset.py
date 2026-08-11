"""
Script de préparation du data set pour évaluation/finetuning
70% du corpus sert à l'entraînement (train)
15% du corpus sert à l'évaluation (test)
15% du corpus est le témoin (dev)
dev et test sont à prendre au milieu de l'enregistrement pour éviter les biais liés au début et à la fin d'une conversation
pour ce faire, on prend 45% (train) - 15% (test) - 15% (dev) - 45% (train) dans l'ordre chronologique de l'audio

pour l'évaluation, il faut que les annotations golds soient extraites
en correspondance à la segmentation du corpus audio
"""

from pydub import AudioSegment # pour les opérations sur l'audio
from pympi.Elan import Eaf # pour les opérations sur les transcription eaf
from speach import elan

from pyannote.core import Annotation, Segment # pour des annotations lisibles par par pyannote.metrics
from os import listdir, makedirs
from shutil import copy
from csv import DictWriter
import exploration as expl

# la fonction divise le corpus et génère et renvoie les annotations gold de chaque partie (train, dev et test)
# TODO: si exacte est vrai: on coupe les tours de paroles pour la segmentation, sinon
# on fait en sorte que les frontièrs de sous-corpus soient allignées aux frontières de tours de parole
def segmenter_corpus(audio_files_path: str, eaf_files_path: str, taille_train: float = .7, taille_test: float = .15, taille_dev: float = .15, exacte: bool = False):
    # on détermine la destination du corpus une fois ségmenté
    direction_train: str = "./corpus/train/"
    direction_test: str = "./corpus/test/"
    direction_dev: str = "./corpus/dev/"
    # et on les crée
    makedirs(direction_train + "audio/", exist_ok=True)
    makedirs(direction_train + "gold/", exist_ok=True)
    makedirs(direction_train + "annotation/", exist_ok=True)
    makedirs(direction_test + "audio/", exist_ok=True)
    makedirs(direction_test + "gold/", exist_ok=True)
    makedirs(direction_test + "annotation/", exist_ok=True)
    makedirs(direction_dev + "audio/", exist_ok=True)
    makedirs(direction_dev + "gold/", exist_ok=True)
    makedirs(direction_dev + "annotation/", exist_ok=True)

    # on importe tous les fichiers wav
    audio_filepath_list: list[str] = [f for f in listdir(audio_files_path) if ".wav" in f]
    transcription_file_list: list[str] = [f for f in listdir(eaf_files_path) if ".eaf" in f]

    enregistrement_annotation: dict[str, tuple] = {}

    for i in audio_filepath_list:
        nom: str = i[:5]
        for j in transcription_file_list:
            if nom == j[:5]:
                enregistrement_annotation[nom] = (j, AudioSegment.from_file(audio_files_path + "/" + i))
        
    # on divise chaque audio en 45% (train) - 15% (test) - 15% (dev) - 45% (train)
    for nom, enregistrement in enregistrement_annotation.items():
        if exacte:
            # on détermine le début et la fin de chaque partie
            train1_début: int = 0
            train1_fin: int = int(len(enregistrement[1]) * (taille_train / 2))

            test_début: int = train1_fin
            test_fin: int = test_début + int(len(enregistrement[1]) * taille_test)

            dev_début: int = test_fin
            dev_fin: int = dev_début + int(len(enregistrement[1]) * taille_dev)

            train2_début: int = dev_fin
            train2_fin: int = len(enregistrement[1])
        else:
            transcriptions_eaf_entière = elan.read_eaf(eaf_files_path + "/" + enregistrement[0])
            
            train1_début, train1_fin = extraire_segment_eaf_timestamps(transcriptions_eaf_entière, 0, int(len(enregistrement[1]) * (taille_train / 2)))
            
            test_début, test_fin = extraire_segment_eaf_timestamps(transcriptions_eaf_entière, train1_fin, train1_fin + int(len(enregistrement[1]) * taille_test))

            dev_début, dev_fin = extraire_segment_eaf_timestamps(transcriptions_eaf_entière, test_fin, test_fin + int(len(enregistrement[1]) * taille_dev))

            train2_début, train2_fin = extraire_segment_eaf_timestamps(transcriptions_eaf_entière, dev_fin, len(enregistrement[1]))

        # on découpe les audios et les enregistre
        ## train
        enregistrement[1][train1_début:train1_fin].set_frame_rate(16_000).export(direction_train + "audio/" + nom + "-train1.wav", format="wav")
        enregistrement[1][train2_début:train2_fin].set_frame_rate(16_000).export(direction_train + "audio/" + nom + "-train2.wav", format="wav")
        print(f"Section train de l'enregistrement audio {nom} a été enregistré dans {direction_train}audio/{nom}-train.wav")
        ## test
        enregistrement[1][test_début:test_fin].set_frame_rate(16_000).export(direction_test + "audio/" + nom + "-test.wav", format="wav")
        print(f"Section test de l'enregistrement audio {nom} a été enregistré dans {direction_test}audio/{nom}-test.wav")
        ## dev
        enregistrement[1][dev_début:dev_fin].set_frame_rate(16_000).export(direction_dev + "audio/" + nom + "-dev.wav", format="wav")
        print(f"Section dev de l'enregistrement audio {nom} a été enregistré dans {direction_dev}audio/{nom}-dev.wav")

        transcriptions_eaf_entière = Eaf(eaf_files_path + "/" + enregistrement[0])

        # on découpe les annotations gold et on les enregistre
        transcriptions_eaf_entière = Eaf(eaf_files_path + "/" + enregistrement[0])
        
        # train
        gold_train1_eaf = extraire_segment_eaf(transcriptions_eaf_entière, train1_début, train1_fin)
        gold_train2_eaf =  extraire_segment_eaf(transcriptions_eaf_entière, train2_début, train2_fin)
        
        gold_train1_eaf.to_file(f"{direction_train}gold/{nom}-train1.eaf")
        print(f"Annotations gold de train1 enregistrées au format eaf dans {direction_train}gold/")
        gold_train2_eaf.to_file(f"{direction_train}gold/{nom}-train2.eaf")
        print(f"Annotations gold de train2 enregistrées au format eaf dans {direction_train}gold/")
        # test
        gold_test_eaf = extraire_segment_eaf(transcriptions_eaf_entière, test_début, test_fin)

        gold_test_eaf.to_file(f"{direction_test}gold/{nom}-test.eaf")
        print(f"Annotations gold de test enregistrées au format eaf dans {direction_test}gold/")
        # dev
        gold_dev_eaf = extraire_segment_eaf(transcriptions_eaf_entière, dev_début, dev_fin)

        gold_dev_eaf.to_file(f"{direction_dev}gold/{nom}-dev.eaf")
        print(f"Annotations gold de dev enregistrées au format eaf dans {direction_dev}gold/")

        # on génère un fichier statistique pour la segmentation actuelle (pour test, train 1&2 et dev)
        ## train 1&2
        expl.explorer("./corpus/train/gold", "./corpus/train/statistiques")
        ## test
        expl.explorer("./corpus/test/gold", "./corpus/test/statistiques")
        ## dev
        expl.explorer("./corpus/dev/gold", "./corpus/dev/statistiques")

def générer_rttm(chemin_eaf: str, chemin_audio: str, chemin_sortie: str, inclure_extra: bool = False):
    # on extrait les annotations de l'eaf pour générer la majorité du rttm, le fichier audio est là car une référence y est nécessaire dans le rttm
    eaf = Eaf(chemin_eaf)

    data: list[str] = []

    for t in [tier for tier in eaf.get_tier_names() if ("EXTRA" not in tier and "extra" not in tier) or inclure_extra]:
        for début, fin, _ in eaf.get_annotation_data_for_tier(t):
            durée = (fin - début) / 1000

            ligne: str = f"SPEAKER {chemin_audio.split('.')[0]} 1 {début/1000} {durée} <NA> <NA> {t} <NA> <NA>\n"

            data.append(ligne)
    
    # on enregistre tout
    with open(chemin_sortie + "/" + chemin_audio.split("/")[-1] + ".rttm", "w") as fichier_rttm:
        fichier_rttm.writelines(data)

def créer_dataset_entrainement_pyannote(location_audio: str, location_gold: str, location_sortie, prendre_extra: bool = False):
    dict_enregistrements: dict[str, dict[str, str]] = {}

    for f in listdir(location_audio):
        nom = f[:5]

        if not nom in dict_enregistrements.keys():
            dict_enregistrements[nom] = {}
        
        dict_enregistrements[nom] = {"audio": f, "gold": ""}
    
    for f in listdir(location_gold):
        nom = f[:5]

        dict_enregistrements[nom]["gold"] = f

    data = {}
    
    makedirs(location_sortie, exist_ok=True)
    for nom, données in dict_enregistrements.items():
        # chaque ligne du fichier csv correspond à un enregistrement
        # la première colonne contient le lien vers l'audio
        nom = données["audio"][:5]

        data[nom] = {"file_name": nom + ".wav", "timestamps_start": [], "timestamps_end": [], "speakers": []}

        # on charge le fichier annotation gold
        fichier_eaf_complet = Eaf(location_gold + "/" + données["gold"])

        # on enregistre les annotations
        for t in [tier for tier in fichier_eaf_complet.get_tier_names() if ("EXTRA" not in tier and "extra" not in tier) or prendre_extra]:
            for début, fin, _ in fichier_eaf_complet.get_annotation_data_for_tier(t):
                début /= 1000
                fin /= 1000

                data[nom]["timestamps_start"].append(début)
                data[nom]["timestamps_end"].append(fin)
                data[nom]["speakers"].append(t)
            print(len(data[nom]["timestamps_start"]) == len(data[nom]["timestamps_end"]) and len(data[nom]["timestamps_start"]) == len(data[nom]["speakers"]))

        # on envoie les fichiers audio dans le dossier dataset pour que
        AudioSegment.from_file(location_audio + "/" + données["audio"], "wav").export(location_sortie + "/" + nom + ".wav", "wav")


    # on enregistre les données dans un csv
    with open(location_sortie + "/metadata" + ".csv", "w", newline='') as fichier:
        csv_writer = DictWriter(fichier, fieldnames=data["GB_E1"].keys(), delimiter=",")
        csv_writer.writeheader()

        for nom, données in data.items():
            csv_writer.writerow(données)

def créer_dataset_entrainement_pyannote2(location_audio: str, location_gold: str, location_sortie, prendre_extra: bool = False):
    # on enregistre d'abord tous les audios dans audio
    for f in [file for file in listdir(location_audio) if ".wav" in file]:
        copy(location_audio + "/" + f, location_sortie + "/audio")

    # on transforme les eaf en rttm et on les ajoute au dataset
    for f in [file for file in listdir(location_gold) if ".eaf" in file]:
        générer_rttm(location_gold + "/" + f, f.split(".")[0], location_sortie + "/annotations")

# extraction de début/fin des sous-corpus sans coupure de tour de parole
def extraire_segment_eaf_timestamps(eaf: elan.Doc, début: int, fin: int):
    for tier in eaf:
        if "EXTRA" not in tier.ID and "extra" not in tier.ID:
            ann_interval = [ann for ann in tier.filter(elan.TimeSlot(value=début), elan.TimeSlot(value=fin))]

            if ann_interval != []:
                début = min(ann_interval[0].from_ts.value, début)
                fin = max(ann_interval[-1].to_ts.value, fin)
    
    return début, fin

# la méthode Eaf.extract(start, end) ne fonctionne de façon attendue:
# plutôt que de créer un nouvel objet Eaf qui commence à start et fini à end, elle créer un objet eaf commençant à 0 et finissant à end
# MAIS les annotations entre 0 et start ne sont pas extraites
# on peut donc créer un nouveau objet Eaf qui récupère toutes les annotations, sans prednre l'espace vide entre 0 et start
def extraire_segment_eaf(eaf: Eaf, début: int, fin: int) -> Eaf:
    eaf = eaf.extract(début, fin)

    nouvel_eaf: Eaf = Eaf()

    for tier in eaf.get_tier_names():
        nouvel_eaf.add_tier(tier)

        for (début_i, fin_i, _) in eaf.get_annotation_data_for_tier(tier):
            if début_i > début and fin_i < fin:
                nouvel_eaf.add_annotation(tier, début_i - début, fin_i - début)
    
    nouvel_eaf.clean_time_slots()
    
    return nouvel_eaf

# fonction qui prend un objet Eaf et retourne un objet Annotation correspondant
# si une liste d'Eaf est donné, on les traite comme différente partie d'un même segment
# permet la concaténation de segment d'annotation non contigues
## TODO: les annotations concaténées ne sont pas correctes: chaque eaf reprend le décompte des annotations à 0s, ce qui n'est l'effet désiré
### à voir comment le corriger plus tard
def extraire_annotation_eaf(eaf_file: Eaf | list[Eaf], compter_extra: bool = False, est_gold: bool = False) -> Annotation:
    annotation = Annotation()

    match eaf_file:
        case [*eafs]:
            for eaf in eafs:
                tier_data: list = []

                if compter_extra:
                    tier_data = eaf.get_tier_names()
                else:
                    tier_data = [t for t in eaf.get_tier_names() if (not "EXTRA" in t and not "extra" in t)]

                for tier in tier_data:
                    for (début, fin, value) in eaf.get_annotation_data_for_tier(tier):
                        if value != "" and est_gold:
                            annotation[Segment(début/1000, fin/1000)] = tier
                        else:
                            annotation[Segment(début/1000, fin/1000)] = tier
        case eaf:
            tier_data: list = []

            if compter_extra:
                tier_data = eaf.get_tier_names()
            else:
                tier_data = [t for t in eaf.get_tier_names() if (not "EXTRA" in t and not "extra" in t)]

            for tier in tier_data:
                for (début, fin, value) in eaf.get_annotation_data_for_tier(tier):
                    if value != "" and est_gold: # les timeslot qui n'ont pas d'annotation représentent, dans le gold, des time slot parasite
                        annotation[Segment(début/1000, fin/1000)] = tier
                    else:
                        annotation[Segment(début/1000, fin/1000)] = tier
    return annotation


def main():
    segmenter_corpus("./corpus/audios", "./corpus/transcriptions", exacte=False)

    créer_dataset_entrainement_pyannote2("./corpus/train/audio", "./corpus/train/gold", "./corpus/dataset3")
    créer_dataset_entrainement_pyannote2("./corpus/test/audio", "./corpus/test/gold", "./corpus/dataset3")
    créer_dataset_entrainement_pyannote2("./corpus/dev/audio", "./corpus/dev/gold", "./corpus/dataset3")

    #générer_rttm("./corpus/train/gold/GC_E1-train1.eaf", "GC_E1", "./corpus")
            


if __name__ == "__main__":
    main()
