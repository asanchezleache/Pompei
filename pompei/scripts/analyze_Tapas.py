"""
Copyright (c) Institut national de l'information géographique et forestière https://www.ign.fr/

File main authors:
- Célestin Huet

This file is part of Pompei: https://github.com/IGNF/Pompei

Pompei is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
Pompei is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with Pompei. If not, see <https://www.gnu.org/licenses/>.
"""

import argparse
import os
import shutil
import log # Chargement des configurations des logs
import logging


logger = logging.getLogger()

parser = argparse.ArgumentParser(description="Analyse du rapport de Tapas ou de Campari. Pour Tapas, on vérifie qu'il n'y a pas de problèmes lors du calcul de l'orientation relative. Si des images posent problème, alors on les supprime et on relance Tapas.")
parser.add_argument('--input_report', help='Rapport Tapas ou Campari')
parser.add_argument('--scripts_dir', help='Répertoire des scripts', default=None)
parser.add_argument('--ta', help="HistCat: fichier TA du chantier (caméras et images, pour la focale initiale)", default=None)
parser.add_argument('--staged_first', action="store_true",
                    help="HistCat: Tapas en deux étapes dès le départ (chantier marqué HISTCAT_tapas_staged.txt)")
args = parser.parse_args()


# HistCat (2026-10-20, run_ledger.md problems 19 and 44-57). Pompei starts every camera at F = 150 mm / 0.021 mm =
# 7,143 px (its default scan pixel when a flight is not in export_focales.json); the Vuelo Nacional's calibrated
# focals are 8,700-10,400 px. On some sheets the plain Tapas then fails: NaN once it frees the focal (Getafe,
# Salamanca), or a wrong Martini start (Logroño). The loop below then removed images in file order (when every
# residual is NaN the "worst" image is just the first of the report) until Tapas converged: Getafe lost 27 of 43
# frames and crashed, Salamanca 7, San Sebastián 5. Fallback: on a FATAL plain Tapas, the "staged Tapas" -- each
# camera's starting focal from its photos' measured side (photo_frame.detect, x 152.5 mm / 230 mm), Martini from it,
# Tapas with the focal fixed (LibFoc=0), then everything free from that solution -- before any image is removed.
# Tested on the saved tie points: Getafe 43/43, Salamanca 43/43; Vigo (plain run fine) gives the same solution
# within 0.1 %. Sheets whose plain Tapas converges never enter this path: their results are unchanged. A chantier
# marked HISTCAT_tapas_staged.txt at staging (plain Tapas known to fail) skips the plain attempt: same result
# (Martini and Tapas are deterministic), minutes saved.
HISTCAT_STAGED = "HISTCAT_tapas_staged.txt"
FOCALE_MM, CADRE_MM = 152.5, 230.0        # RC-10 / RMK lens, 23 cm frame
PART_PHOTO = 0.90                          # fallback: the photo is ~90 % of the scan width
TAPAS_ETAPE1 = ("mm3d Tapas FraserBasic \"OIS-Reech_.*tif\" InOri=MartiniCalibHC InCal=CalibHC LibFoc=0 Out=RelHC0 "
                "@ExitOnWarn | tee reports/report_Tapas_etape1.txt >> logfile")
TAPAS_ETAPE2 = ("mm3d Tapas FraserBasic \"OIS-Reech_.*tif\" InOri=RelHC0 InCal=RelHC0 Out=Rel @ExitOnWarn "
                "| tee reports/report_Tapas.txt >> logfile")


def report_fatal(chemin_rapport):
    """True if a Tapas run failed: its report ends on a fatal error, on a warning that @ExitOnWarn turned into an exit,
    or with a NaN residual (San Sebastián, Palma: the plain run ended at "Residual = -nan" without a FATAL message, and
    the loop then removed images), or is missing."""
    if not os.path.exists(chemin_rapport):
        return True
    with open(chemin_rapport, "r", errors="replace") as f:
        texte = f.read()
    residus = [l for l in texte.splitlines() if "Residual =" in l]
    return ("Sorry, the following FATAL ERROR happened" in texte or "Warn tape enter to continue" in texte
            or bool(residus and "nan" in residus[-1].split("Residual =")[1].split(";")[0].lower()))


def cameras_images(chemin_ta):
    """Images of each camera, in Pompei's numbering (tools.getSensors: the TA vols whose images are present)."""
    import glob
    from lxml import etree
    if chemin_ta is None or not os.path.exists(chemin_ta):
        tas = [t for t in glob.glob("TA*.xml") if "updated" not in t]
        chemin_ta = tas[0] if len(tas) == 1 else None
    if chemin_ta is None:
        return []
    cameras = []
    for vol in etree.parse(chemin_ta).getroot().iter("vol"):
        images = [f"OIS-Reech_{c.find('image').text.strip()}.tif" for c in vol.iter("cliche")]
        images = [i for i in images if os.path.exists(i)]
        if images:
            cameras.append(images)
    return cameras


def focales_initiales(chemin_ta):
    """Ori-CalibHC: Ori-CalibNum with each camera's focal = median photo side (px) x 152.5 / 230, the photo found by
    photo_frame.detect on up to 5 of the camera's images (none found: 90 % of the scan width). Computed once."""
    import glob
    import re
    from statistics import median
    if os.path.isdir("Ori-CalibHC"):
        return
    import photo_frame
    cameras = cameras_images(chemin_ta)
    os.system("rm -rf Ori-CalibHC && cp -r Ori-CalibNum Ori-CalibHC")
    for fichier in sorted(glob.glob("Ori-CalibHC/*.xml")):
        texte = open(fichier).read()
        m = re.search(r"Argentique(\d+)", fichier)
        cam = int(m.group(1)) if m else 0
        cotes = []
        images = cameras[cam] if cam < len(cameras) else []
        for image in images[:: max(1, len(images) // 5)][:5]:
            small, _, _ = photo_frame.read_small(image)
            det = photo_frame.detect(small)
            c0, r0, c1, r1 = det["bbox"]          # full-resolution pixels, end exclusive (whole image: nothing found)
            if (c0, r0) != (0, 0) or (c1, r1) != (small.shape[1] * det["step"], small.shape[0] * det["step"]):
                cotes.append(((c1 - c0) + (r1 - r0)) / 2)
        if cotes:
            f0, source = median(cotes) * FOCALE_MM / CADRE_MM, f"photo side ~{median(cotes):.0f} px on {len(cotes)} images"
        else:
            largeur = float(re.search(r"<SzIm>\s*(\d+)", texte).group(1))
            f0, source = PART_PHOTO * largeur * FOCALE_MM / CADRE_MM, f"no photo found: 90 % of the scan width {largeur:.0f}"
        texte = re.sub(r"<F>[^<]*</F>", f"<F>{f0:.1f}</F>", texte)
        open(fichier, "w").write(texte)
        logger.info(f"HISTCAT TAPAS: {os.path.basename(fichier)}: starting focal {f0:.1f} px ({source})")


def staged_tapas(chemin_ta):
    """Martini and Tapas from the starting focals of focales_initiales(): focal fixed, then everything free from that
    solution. Writes Ori-Rel and reports/report_Tapas.txt as the plain run does. If the release fails, the fixed-focal
    solution is kept (aero's Campari frees every parameter with the anchors). If the fixed stage itself fails, its
    report becomes report_Tapas.txt, so the image-removal loop takes over (with this staged Tapas)."""
    focales_initiales(chemin_ta)
    os.system("rm -rf Ori-Rel Ori-RelHC0")
    os.system("mm3d Martini \"OIS-Reech_.*tif\" OriCalib=CalibHC >> logfile")
    os.system(TAPAS_ETAPE1)
    if report_fatal("reports/report_Tapas_etape1.txt"):
        logger.warning("HISTCAT TAPAS: staged Tapas, focal fixed: failed")
        shutil.copy("reports/report_Tapas_etape1.txt", "reports/report_Tapas.txt")
        return False
    os.system(TAPAS_ETAPE2)
    if report_fatal("reports/report_Tapas.txt"):
        logger.warning("HISTCAT TAPAS: staged Tapas: releasing the focal failed; the fixed-focal solution is kept")
        os.system("rm -rf Ori-Rel && cp -r Ori-RelHC0 Ori-Rel && cp reports/report_Tapas_etape1.txt reports/report_Tapas.txt")
    else:
        logger.info("HISTCAT TAPAS: staged Tapas converged (focal fixed, then free)")
    return True



def remove_images_without_homol(dictionnaire):
    images = [i for i in os.listdir() if i[-4:]==".tif" and i[:9]=="OIS-Reech"]
    for image in images:
        if image not in dictionnaire.keys():
            logger.warning("On retire l'image {} car elle n'a plus de points de liaisons avec des images encore existantes".format(image))
            shutil.move(image, "Poubelle_Tapas")



def find_problem(chemin_rapport, chemin_scripts):
    dictionnaire = {}

    line_worst = None
    line_residual = None   # HistCat fix: was unbound when Tapas crashes early

    with open(chemin_rapport, "r") as f:
        fatal_error = False
        for line in f:
            if "RES" in line and "ER2" in line:
                # Si c'est une ligne avec les informations concernant les résidus d'une image
                # nom de l'image
                line_replaced = line.replace("]", "[")
                line_splitted = line_replaced.split("[")
                nom_image = line_splitted[1]

                # proportion de points de liaisons conservés
                line_splitted_space = line.split()
                proportion = line_splitted_space[4]
                if proportion == "-nan":
                    proportion = 0


                # Résidus
                residus = line_splitted_space[2]
                dictionnaire[nom_image] = {"residus":residus, "proportion":proportion}

            if "NON INIT" in line:
                line_splitted = line.split(" ")
                nom_image = line_splitted[6][:-1]
                dictionnaire[nom_image] = {"residus":0, "proportion":0}
            
            # Ligne avec les résidus moyens
            if "Residual =" in line:
                line_residual = line
            # Ligne avec les moins bons résultats 
            if "Worst" in line:
                line_worst = line
            
            # Erreurs diverses
            if "Sorry, the following FATAL ERROR happened" in line:
                logger.warning(f"{line}")
                fatal_error = True
            if "Warn tape enter to continue" in line:
                logger.warning(f"{line}")
                fatal_error = True

    if "Tapas" in os.path.basename(chemin_rapport):
        algo="Tapas"
    else:
        algo = "Campari"

    
    # On récupère l'image qui a la proportion de points de liaisons conservés la plus faible
    image_min = None
    val_min = 100.00
    for image in dictionnaire:
        logger.debug(f"Image : {image}, points de liaisons conservés : {dictionnaire[image]['proportion']}, résidus : {dictionnaire[image]['residus']}")
        if float(dictionnaire[image]["proportion"]) < val_min:
            val_min = float(dictionnaire[image]["proportion"])
            image_min = image
    
    logger.info(line_residual)
    logger.info(line_worst)

    # HistCat: a failed plain Tapas (FATAL, or ending at a NaN residual) goes to the staged Tapas before any image is
    # removed (see HISTCAT_STAGED)
    if algo=="Tapas" and report_fatal(chemin_rapport) and not os.path.exists(HISTCAT_STAGED):
        with open(HISTCAT_STAGED, "w") as f:
            f.write("staged Tapas after a fatal plain Tapas (fallback)\n")
        logger.warning("HISTCAT TAPAS: the plain Tapas failed: staged Tapas (starting focal from the photos, focal fixed, then free)")
        staged_tapas(args.ta)
        os.system("python {}/analyze_Tapas.py --input_report reports/report_Tapas.txt --scripts_dir {}{}".format(
            chemin_scripts, chemin_scripts, f" --ta {args.ta}" if args.ta else ""))
        return

    # Si on est dans le premier calcul d'orientation relative
    if algo=="Tapas":
        recommencer_Tapas = False
        # Si le calcul n'a pas abouti ou que la proportion minimale de points de liaisons utilisés est inférieure à 60, on supprime la moins bonne image
        if fatal_error or val_min <= 60.0:
            recommencer_Tapas = True
            logger.warning("L'image {} est retirée du jeu de données car Tapas n'est pas parvenu à déterminer son orientation relative".format(image_min))
            if not os.path.exists("Poubelle_Tapas"):
                os.mkdir("Poubelle_Tapas")
            shutil.move(image_min, "Poubelle_Tapas")
        
        # Supprime les images qui ne sont pas apparues dans le rapport de Tapas car elle n'ont pas de points de liaisons avec les autres images
        remove_images_without_homol(dictionnaire)   
        
        # Si une image a été supprimée, alors on recommence le calcul (sauf si l'image n'apparaît pas dans le rapport Tapas et dans ce cas elle n'avait déjà pas été prise en compte dans le calcul)
        if recommencer_Tapas:
            logger.info("On recommence un calcul d'orientation relative avec Tapas")     
            os.system("rm -r Ori-Rel")
            os.system("rm -r reports/report_Tapas.txt")
            if os.path.exists(HISTCAT_STAGED):      # HistCat: this chantier uses the staged Tapas
                staged_tapas(args.ta)
            else:
                os.system("mm3d Martini \"OIS-Reech_.*tif\" OriCalib=CalibNum >> logfile")
                os.system("mm3d Tapas FraserBasic \"OIS-Reech_.*tif\" InOri=MartiniCalibNum InCal=CalibNum Out=Rel @ExitOnWarn | tee reports/report_Tapas.txt >> logfile")
            os.system("python {}/analyze_Tapas.py --input_report reports/report_Tapas.txt --scripts_dir {}{}".format(
                chemin_scripts, chemin_scripts, f" --ta {args.ta}" if args.ta else ""))

    


if __name__ == "__main__":

    if args.staged_first:     # HistCat: chantier marked HISTCAT_tapas_staged.txt at staging
        logger.info("HISTCAT TAPAS: chantier marked for the staged Tapas: no plain attempt")
        staged_tapas(args.ta)
    find_problem(args.input_report, args.scripts_dir)


    