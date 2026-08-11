from os import listdir, makedirs
from pympi.Elan import Eaf

import torch


def diariser_nemo(chemin_audio: str, chemin_sortie: str, hugging_face_token: str, format_sortie: str = "eaf"):
    from nemo.collections.asr.models import SortformerEncLabelModel
    
    modèle_diarisation = SortformerEncLabelModel.from_pretrained("nvidia/diar_streaming_sortformer_4spk-v2")
    modèle_diarisation.eval()

    makedirs(chemin_sortie + "/" + format_sortie, exist_ok=True)

    diarisation = None
    for f in listdir(chemin_audio):
        if ".wav" in f:
            diarisation = modèle_diarisation.diarize(audio=chemin_audio+"/"+f, batch_size=1)
        
        nom_audio: str = f[:5]
        eaf_i: Eaf = Eaf()
        for segment in diarisation[0]:
            début, fin, locuteur = segment.split(" ")

            # convertit les secondes de nemo en ms pour le fichier Eaf
            début = float(début) * 1000
            fin = float(fin) * 1000

            # on enlève les virgules (qui normalement sont 0)
            début = int(début)
            fin = int(fin)

            try:
                eaf_i.add_annotation(locuteur, début, fin)
            except KeyError: # si on a pas encore ajouter le locuteur tier, on le fait, puis on continue d'ajouter les annotations
                eaf_i.add_tier(locuteur)
                eaf_i.add_annotation(locuteur, début, fin)

        eaf_i.to_file(chemin_sortie + "/" + format_sortie + "/" + nom_audio + ".eaf")


def diariser_pyannote(chemin_audio: str, chemin_sortie: str, hugging_face_token: str, format_sortie: str = "eaf", custom_model: str = ""):
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.utils.hook import ProgressHook
    from diarizers import SegmentationModel

    # on créé le dossier de sortie
    makedirs(chemin_sortie + "/" + format_sortie, exist_ok=True)

    # on charge la pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=hugging_face_token)
    pipeline.to(torch.device("cuda"))
    
    if custom_model:
    # on charge le modèle custom
       modèle = SegmentationModel().from_pretrained(custom_model)
       modèle = modèle.to_pyannote_model()
       pipeline._segmentation.model = modèle.to("cuda")

    # on fait les inférences par fichier audio
    for f in listdir(chemin_audio):

        if not ".wav" in f:
            continue

        with ProgressHook() as hook:
            diarisation = pipeline(chemin_audio+"/"+f, hook=hook)
        
        nom_audio: str = f[:5]

        # on enregistre au format eaf
        if format_sortie == "eaf":
            eaf_i: Eaf = Eaf()
            for tour, locuteur in diarisation.speaker_diarization:
                début: int = int(tour.start * 1000)
                fin: int = int(tour.end * 1000)

                try:
                    eaf_i.add_annotation(locuteur, début, fin)
                except KeyError:
                    eaf_i.add_tier(locuteur)
                    eaf_i.add_annotation(locuteur, début, fin)

            eaf_i.to_file(chemin_sortie + "/" + format_sortie + "/" + nom_audio + ".eaf")
        # on enregistre au format RTTM
        elif format_sortie == "rttm":
            with open(chemin_audio + "/" + format_sortie + "/" + nom_audio + ".rttm") as fichier_sortie:
                for tour, locuteur in diarisation.speaker_diarization:
                    durée_tour = tour.fin - tour.fin
                    
                    fichier_sortie.wirte(f"SPEAKER {f.split(".")[0]} 1 {tour.start} {durée_tour} <NA> <NA> {locuteur} <NA> <NA>\n")


"""
diariser_pyannote(
    "./corpus/dev/audio", 
    "./corpus/dev/annotations-pyannote-finetunées", 
    "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",
    custom_model="./models/pyannote_segmentation_fine_tuné_données_CAENNAIS"
)
"""
diariser_nemo(
    "./corpus/dev/audio", 
    "./corpus/dev/nemo", 
    "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"
)