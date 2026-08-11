from pympi.Elan import Eaf # pour les opérations sur les transcription eaf
import csv
from os import listdir, makedirs
import itertools

# la fonction d'exploration
def explorer(dossier_eafs: str, chemin_sortie: str, inclut_extra: bool = False):
    makedirs(chemin_sortie, exist_ok=True)

    dict_eafs: dict[str, Eaf] = {}

    # on récupère les fichiers eafs des enregistrement à traiter,
    # chaque enregistrement est identifié par les 5 premiers caractères du nom de fichier
    # généralement de la forme GA_E1, etc.
    for f in listdir(dossier_eafs):
        if ".eaf" in f:
            dict_eafs[f[:5]] = Eaf(dossier_eafs+"/"+f)
    
    # on calcul d'abord des stats sur l'intégralité du corpus
    # ici, on récupère la durée totale de chaque enregistrement, pour calculer les %
    durée_totale_corpus: float = sum([x.get_full_time_interval()[1] for x in dict_eafs.values()]) # ms

    # stats par groupes
    données_groupes: dict[str, dict[str, ]] = {}
    ## on ajoute les id de groupes (de la forme GA, GB, GC, etc)
    for g, données in dict_eafs.items():
        nom = g[:2]

        # le nom des différentes mesures est stocké dans une dictionnaire
        if nom not in données_groupes.keys():
            données_groupes[nom] = {
                "durée_totale": 0,
                "nb_tour_parole": 0,
                "durée_moyenne_tour": 0.0
            }

        # on calcul la durée totale de tous les enregistrements de chaque groupe
        données_groupes[nom]["durée_totale"] += données.get_full_time_interval()[1]
        # ainsi que le nombre de tour de parole (permet de savoir quelle loc parle le plus dans son groupe)
        données_groupes[nom]["nb_tour_parole"] += sum([len(données.get_annotation_data_for_tier(tier)) for tier in données.get_tier_names() if ("EXTRA" not in tier and "extra" not in tier) or inclut_extra])
    
    # on calcul la durée moyenne du tour de parole dans chaque groupe
    for groupe, données in données_groupes.items():
        données_groupes[groupe]["durée_moyenne_tour"] = données_groupes[groupe]["durée_totale"] / données_groupes[groupe]["nb_tour_parole"]

    # stats par enregistrement, ici on sort du niveau du groupe, on fait les mêmes stats, enregistrement par enregistrement
    # les stats sont calculés de façon très similaire
    #                           groupe    enregi    mesure
    données_enregistrements: dict[str, dict[str, dict[str, ]]] = {}

    for nom, eaf in dict_eafs.items():
        g_nom = nom[:2]
        e_nom = nom[3:5]

        if g_nom not in données_enregistrements.keys():
            données_enregistrements[g_nom] = {}
        
        if e_nom not in données_enregistrements[g_nom].keys():
            données_enregistrements[g_nom][e_nom] = {
                "durée_totale": 0,
                "nb_tour_parole": 0,
                "durée_moyenne_tour": 0
            }
        
        données_enregistrements[g_nom][e_nom]["durée_totale"] = eaf.get_full_time_interval()[1]
        données_enregistrements[g_nom][e_nom]["nb_tour_parole"] = sum([len(eaf.get_annotation_data_for_tier(tier)) for tier in eaf.get_tier_names()])


    #print(données_groupes)
    # on finit par faire les stats par locuteur, encore une fois, de façon similaire, mais par rapport à son groupe
    # i.e. la durée totale de parole de LOC1 donc GA
    #                      groupe      loc       mesure
    données_loc_groupe: dict[str, dict[str, dict[str, ]]] = {}
    # stats par locuteurs
    for g, données in dict_eafs.items():
        g_nom = g[:2]

        if g_nom not in données_loc_groupe.keys():
            données_loc_groupe[g_nom] = {}

        for tier in [t for t in données.get_tier_names() if ("EXTRA" not in t and "extra" not in t) or inclut_extra]:
            if tier not in données_loc_groupe[g_nom].keys():
                données_loc_groupe[g_nom][tier] = {
                    "durée_totale": 0,
                    "nb_tour_parole": 0,
                    "durée_moyenne_tour_groupe": 0.0
                }
            
            données_loc_groupe[g_nom][tier]["durée_totale"] += sum([fin - début for début, fin, _ in données.get_annotation_data_for_tier(tier)])
            données_loc_groupe[g_nom][tier]["nb_tour_parole"] += len(données.get_annotation_data_for_tier(tier))
    
    for g in données_loc_groupe.keys():
        for loc in données_loc_groupe[g].keys():
            if données_loc_groupe[g][loc]["nb_tour_parole"] > 0:
                données_loc_groupe[g][loc]["durée_moyenne_tour_groupe"] = données_loc_groupe[g][loc]["durée_totale"] / données_loc_groupe[g][loc]["nb_tour_parole"]
    
    #print(données_loc_groupe)
    # ici, on prend des mesures de chaque locuteur, localement à chaque enregistrement
    # i.e. la durée totale de parole de LOC1 dans GA_E2, etc.
    #                              groupe    enregi      loc       mesure
    données_loc_enregistrement: dict[str, dict[str, dict[str, dict[str, ]]]] = {}
    for g, eaf in dict_eafs.items():
        g_nom = g[:2]
        e_nom = g[3:5]

        if g_nom not in données_loc_enregistrement.keys():
            données_loc_enregistrement[g_nom] = {}
        
        if e_nom not in données_loc_enregistrement[g_nom].keys():
            données_loc_enregistrement[g_nom][e_nom] = {}
        
        for loc in [t for t in eaf.get_tier_names() if ("EXTRA" not in t and "extra" not in t) or inclut_extra]:
            if loc not in données_loc_enregistrement[g_nom][e_nom].keys():
                données_loc_enregistrement[g_nom][e_nom][loc] = {
                    "durée_totale": 0,
                    "nb_tour_parole": 0,
                    "durée_moyenne_tour_enregistrement": 0.0
                }
            
            données_loc_enregistrement[g_nom][e_nom][loc]["durée_totale"] += sum([fin - début for début, fin, _ in eaf.get_annotation_data_for_tier(loc)])
            données_loc_enregistrement[g_nom][e_nom][loc]["nb_tour_parole"] += len(eaf.get_annotation_data_for_tier(loc))

    données_superposition_enregistrement: dict[str, dict[str, ]] = {}
    # calcul des durées de paroles superposées par paire de locuteur et nombre d'instance de paroles superposés
    # et des pauses inter et intra locuteurs
    # la méthode utilisée est décrite dans cet article: 
    # Heldner, M., & Edlund, J. (2010). Pauses, gaps and overlaps in conversations. Journal of Phonetics, 38(4), 555–568. doi:10.1016/j.wocn.2010.08.002
    # pour plus d'info concernant les retours de la fonction get_gaps_and_overlaps2: https://dopefishh.github.io/pympi/Elan.html#pympi.Elan.Eaf.get_gaps_and_overlaps2
    for nom, eaf in dict_eafs.items():
        e_nom = nom[:5]

        if e_nom not in données_superposition_enregistrement.keys():
            données_superposition_enregistrement[e_nom] = {
                "durée_totale_superpo": 0,
                "nb_instance_superpo": 0,
                "nb_pause_intra": 0,
                "durée_totale_pause_intra": 0,
                "nb_pause_inter": 0,
                "durée_totale_pause_inter": 0,
            }

        liste_combinaison_loc_unique: list(tuple) = list(itertools.combinations(eaf.get_tier_names(), 2))

        for loc1, loc2 in liste_combinaison_loc_unique:
            liste_pause_intralocuteur = [p_intra for p_intra in eaf.get_gaps_and_overlaps2(loc1, loc2) if p_intra[2][0] == "P"]
            liste_pause_inter_locuteur = [p_inter for p_inter in eaf.get_gaps_and_overlaps2(loc1, loc2) if p_inter[2][0] == "G"]
            liste_overlap = [o for o in eaf.get_gaps_and_overlaps2(loc1, loc2) if o[2][0] == "O" or o[2][0] == "W"]
            
            données_superposition_enregistrement[e_nom]["durée_totale_superpo"] += sum([fin - début for début, fin, _ in liste_overlap])
            données_superposition_enregistrement[e_nom]["nb_instance_superpo"] += len(liste_overlap)

            données_superposition_enregistrement[e_nom]["nb_pause_intra"] += len(liste_pause_intralocuteur)
            données_superposition_enregistrement[e_nom]["durée_totale_pause_intra"] += sum([fin - début for début, fin, _ in liste_pause_intralocuteur])

            données_superposition_enregistrement[e_nom]["nb_pause_inter"] += len(liste_pause_inter_locuteur)
            données_superposition_enregistrement[e_nom]["durée_totale_pause_inter"] += sum([fin - début for début, fin, _ in liste_pause_inter_locuteur])
    
    # print(données_superposition_enregistrement)

    
    # fin des mesures, passage aux calculs statistiques et à l'enregitrement
    ## durée de parole de Loc dans Gr par rapport à Loc dans E
    for nom, eaf in dict_eafs.items():
        nom_fichier_stat: str = ""

        if inclut_extra: nom_fichier_stat = chemin_sortie + "/" + nom + "_extra.csv"
        else: nom_fichier_stat = chemin_sortie + "/" + nom + "_noExtra.csv"

        with open(nom_fichier_stat, "w") as fichier:
            csv_writer = csv.writer(fichier)

            e_nom = nom[3:5]
            g_nom = nom[:2]
            
            csv_writer.writerow([
                "Durée totale de l'enregistrement (s)",
                str(données_enregistrements[g_nom][e_nom]["durée_totale"] / 1000)
            ])
            csv_writer.writerow([
                "Nombre totale de tour de paroles (tdp)",
                str(données_enregistrements[g_nom][e_nom]["nb_tour_parole"])
            ])
            csv_writer.writerow([
                "Durée moyenne du tour de parole",
                str((données_enregistrements[g_nom][e_nom]["durée_totale"] / 1000) / données_enregistrements[g_nom][e_nom]["nb_tour_parole"])
            ])
            csv_writer.writerow([
                "Nombre d'instance de parole superposée dans l'enregistrement",
                str(données_superposition_enregistrement[nom]["nb_instance_superpo"])
            ])
            csv_writer.writerow([
                "Durée totale de parole superposée dans l'enregistrement (s)",
                str(données_superposition_enregistrement[nom]["durée_totale_superpo"] / 1000)
            ])
            csv_writer.writerow([
                "Proportion de parole superposée dans l'enregistrement (%)",
                str((données_superposition_enregistrement[nom]["durée_totale_superpo"] / données_enregistrements[g_nom][e_nom]["durée_totale"]) * 100)
            ])
            csv_writer.writerow([
                "Nombre de pauses intra-locuteur dans l'enregistrement",
                str(données_superposition_enregistrement[nom]["nb_pause_intra"])
            ])
            csv_writer.writerow([
                "Durée totale des pauses intra-locuteur dans l'enregistrement (s)",
                str(données_superposition_enregistrement[nom]["durée_totale_pause_intra"] / 1000)
            ])
            csv_writer.writerow([
                "Durée moyenne des pauses intra-locuteur dans l'enregistrement (s)",
                str((données_superposition_enregistrement[nom]["durée_totale_pause_intra"] / 1000) / données_superposition_enregistrement[nom]["nb_pause_intra"])
            ])
            csv_writer.writerow([
                "Nombre de pauses inter-locuteur dans l'enregistrement",
                str(données_superposition_enregistrement[nom]["nb_pause_inter"])
            ])
            csv_writer.writerow([
                "Durée totale des pauses inter-locuteur dans l'enregistrement (s)",
                str(données_superposition_enregistrement[nom]["durée_totale_pause_inter"] / 1000)
            ])
            csv_writer.writerow([
                "Durée moyenne des pauses inter-locuteur dans l'enregistrement (s)",
                str((données_superposition_enregistrement[nom]["durée_totale_pause_inter"] / 1000) / données_superposition_enregistrement[nom]["nb_pause_inter"])
            ])
            
            csv_writer.writerow([
                "",
                "Nombre de tour de parole",
                "Durée totale de parole",
                "Durée moyenne tdp dans l'enregistrement",
                "Durée moyenne tdp dans le groupe",
                "Proportion dans l'enregistrement",
                "Proportion dans le groupe"
            ])

            for loc, données in données_loc_enregistrement[g_nom][e_nom].items():
                try:
                    csv_writer.writerow([
                        loc,
                        données["nb_tour_parole"],
                        données["durée_totale"] / 1000,
                        (données["durée_totale"] / 1000) / données["nb_tour_parole"],
                        (données_loc_groupe[g_nom][loc]["durée_totale"] / 1000) / données_loc_groupe[g_nom][loc]["nb_tour_parole"],
                        (données["durée_totale"] / données_enregistrements[g_nom][e_nom]["durée_totale"]) * 100,
                        (données_loc_groupe[g_nom][loc]["durée_totale"] / données_groupes[g_nom]["durée_totale"]) * 100
                    ])
                except (ZeroDivisionError, IndexError):
                    pass


def main():
    # utilisation exemple, à modifier
    explorer("./corpus/GF_temp", "./corpus/stats", False)

if __name__ == "__main__":
    main()