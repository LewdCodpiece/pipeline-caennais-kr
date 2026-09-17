#from transformers import Trainer, TrainingArguments
from os import listdir, makedirs
import csv

from pydub import AudioSegment
from pympi.Elan import Eaf
from datasets import load_dataset, Audio
from transformers import Trainer


def créer_dateset(chemin_audio: str, chemin_eaf: str, chemin_sortie: str, compter_extra: bool = False):
    audio_annotation: dict[str, dict[str, str]] = {}

    for f in listdir(chemin_audio):
        if ".wav" not in f:
            continue
        
        nom = f[:5]
        audio_annotation[nom] = {}

        audio_annotation[nom]["audio"] = f

    for f in listdir(chemin_eaf):
        if ".eaf" not in f:
            continue
        
        nom = f[:5]
        audio_annotation[nom]["annotation"] = f
    
    data: dict[str, list[tuple]] = {}

    for nom, fichiers in audio_annotation.items():
        data[nom] = []

        audio, eaf = fichiers["audio"], fichiers["annotation"]

        audio_complet = AudioSegment.from_wav(chemin_audio + "/" + audio)
        # on met l'audio entier dans le data aussi
        audio_complet.export(chemin_sortie + "/" + nom + "/" + audio, "wav")
        eaf_complet = Eaf(chemin_eaf + "/" + eaf)

        for tier in eaf_complet.get_tier_names():
            if not compter_extra and ("EXTRA" in tier or "extra" in tier):
                continue

            for début, fin, annotation in eaf_complet.get_annotation_data_for_tier(tier):
                durée = fin - début # ms

                # on garde pas les annotations de plus de 30s
                if durée > 30_000:
                    continue

                extrait_audio = audio_complet[début:fin]
                extrait_audio_chemin: str = chemin_sortie + f"/{nom}/{audio}-{str(début)}-{str(fin)}-{tier}.wav"

                makedirs(chemin_sortie + f"/{nom}", exist_ok=True)
                extrait_audio.export(extrait_audio_chemin, "wav")

                data[nom].append((nom + "/" + audio, début/1000, fin/1000, tier))
    
    with open(chemin_sortie + "/metadata.csv", "w") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")

        csv_writer.writerow(["file_name", "timestamps_start", "timestamps_end", "speakers"])
        for nom in data.keys():
            for ligne in data[nom]:
                csv_writer.writerow(ligne)
                
# en local
def charger_dataset(dataset_chemin: str):
    chemin_test: str = f"{dataset_chemin}/test/csv"
    chemin_train: str = f"{dataset_chemin}/train/csv"

    """
    dataset = load_dataset("csv", data_files={
        "train": [chemin_train + "/" + f for f in listdir(chemin_train) if ".csv" in f],
        "test": [chemin_test + "/" + f for f in listdir(chemin_test) if ".csv" in f]
    })
    """
    dataset = load_dataset("audiofolder", data_dir=dataset_chemin)

    print(dataset)

    return dataset

def charger_dataset2(chemin: str):
    from diarizers import SpeakerDiarizationDataset

    fichiers_annotations: dict[str, list] = {
        "train": [chemin + "/annotations/" + f for f in listdir(chemin + "/annotations") if "train" in f and ".rttm" in f],
        "test": [chemin + "/annotations/" + f for f in listdir(chemin + "/annotations") if "test" in f and ".rttm" in f],
        "dev": [chemin + "/annotations/" + f for f in listdir(chemin + "/annotations") if "dev" in f and ".rttm" in f]
    }

    fichiers_audios: dict[str, list] = {
        "train": [chemin + "/audio/" + f for f in listdir(chemin + "/audio") if "train" in f and ".wav" in f],
        "test": [chemin + "/audio/" + f for f in listdir(chemin + "/audio") if "test" in f and ".wav" in f],
        "dev": [chemin + "/audio/" + f for f in listdir(chemin + "/audio") if "dev" in f and ".wav" in f]
    }

    return SpeakerDiarizationDataset(fichiers_audios, fichiers_annotations).construct_dataset()

def entrainer_nemo():
    pass

def entrainer_pyannote(hf_token: str, chemin_sortie: str = "./modèles/modèles_pyannote_fine_tunés"):
    from pyannote.audio import Model
    from diarizers import Preprocess, SegmentationModel,  DataCollator, Metrics
    
    makedirs(chemin_sortie, exist_ok=True)

    # on charge un checkpoint de community-1
    préentrainé = Model.from_pretrained(
        "pyannote/segmentation-3.0",
        token=hf_token
    )
    model = SegmentationModel.from_pyannote_model(préentrainé)

    préprocesseur = Preprocess(model.config)

    dataset = charger_dataset2("./corpus/dataset3")

    train_set = dataset["train"].map(
        lambda file: préprocesseur(file, random=False, overlap=.5),
        num_proc=4,
        remove_columns=next(iter(dataset.values())).column_names,
        batched=True,
        batch_size=1,
    ).shuffle().with_format("torch")

    eval_set = dataset["test"].map(
        lambda file: préprocesseur(file, random=False, overlap=0.0),
        num_proc=4,
        remove_columns=next(iter(dataset.values())).column_names,
        batched=True, 
        keep_in_memory=True, 
        batch_size=1
    ).with_format("torch")

    metrics = Metrics(model.specifications)

    trainer = Trainer(
        model=model,
        train_dataset=train_set,
        data_collator=DataCollator(max_speakers_per_chunk=model.config.max_speakers_per_chunk),
        eval_dataset=eval_set,
        compute_metrics=metrics
    )

    first_eval = trainer.evaluate()
    print("DER initiale : ", first_eval)
    trainer.train()
    deuxieme_eval = trainer.evaluate()
    print("DER après finetuning : ", deuxieme_eval)
    trainer.save_model(chemin_sortie)

#créer_dateset("./corpus/test/audio", "./corpus/test/gold", "./corpus/dataset/test")
#créer_dateset("./corpus/train/audio", "./corpus/train/gold", "./corpus/dataset/train")
#créer_dateset("./corpus/dev/audio", "./corpus/dev/gold", "./corpus/dataset/dev")

entrainer_pyannote("<huggingface token>")
