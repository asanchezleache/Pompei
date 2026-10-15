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

import os
import json
import argparse
from equations import MNT
import numpy as np
from osgeo import gdal, osr
from multiprocessing import Pool
from tools import getEPSG, load_bbox, getNbCouleurs, getResolution, read_ori
import log # Chargement des configurations des logs
import logging
from tqdm import tqdm
import geopandas as gpd
import rasterio
from rasterio import Affine
import rasterio.features
from shapely import Polygon, intersects, intersection, make_valid

logger = logging.getLogger()

parser = argparse.ArgumentParser(description="Crée un nouveau fichier TA avec les valeurs déterminées pendant le chantier (orientation, position...")

parser.add_argument('--ori', help="Répertoire contenant les fichiers orientations")
parser.add_argument('--cpu', help="Nombre de cpus à utiliser", type=int)
parser.add_argument('--mnt', help="MNT sous format vrt")
parser.add_argument('--radiom', help="Répertoire avec les images égalisées")
parser.add_argument('--mosaic', help="Fichier avec la mosaïque")
parser.add_argument('--ta', help="Fichier TA")
parser.add_argument('--outdir', help="Répertoire contenant les fichiers orientations")
args = parser.parse_args()

ori_path = args.ori
nb_cpus = args.cpu
mnt_path = args.mnt
outdir = args.outdir
radiom = args.radiom
mosaic_file = args.mosaic
ta_path = args.ta

# Une dalle : 2000 pixels
tileSize = 2000


def computeMosaic(x0, y0, shots):
    """
    Pour chaque pixel de la tuile, on cherche le sommet de prise de vue qui lui est le plus proche
    """
    shots_filtered = []
    transform = Affine(resolution, 0, x0, 0, -resolution, y0)
    x_max = x0+2000*resolution
    y_min = y0-2000*resolution
    polygone = Polygon([[x0, y0], [x_max, y0], [x_max, y_min], [x0, y_min]])
    mosaique_extraction = []
    compte = 0
    for element in mosaique:
        if intersects(element[0], polygone):
            try:
                geom_valid = make_valid(element[0])
                geom = intersection(geom_valid, polygone)
            except:
                geom = element[0]
            
            # HistCat (2026-10-15): polygon i is burnt as i+1 and 0 means "no polygon". Before, the first photo was
            # burnt as 0, the fill value, so every pixel outside all polygons took the first photo of the tile.
            mosaique_extraction.append([geom, compte + 1])
            shots_filtered.append(shots[element[1]])
            compte += 1

    if not mosaique_extraction:
        return np.zeros((2000, 2000), dtype=np.uint16), shots_filtered
    mosaic = rasterio.features.rasterize(mosaique_extraction, out_shape=(2000,2000), transform=transform, fill=0,
                                         dtype=np.uint16)
    return mosaic, shots_filtered

    

def saveImage(image, path, x0, y0, EPSG):
    if image.ndim == 2:
        image = image[np.newaxis,:,:]
    nb_bands = image.shape[0]
    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(path, image.shape[2], image.shape[1], nb_bands, gdal.GDT_Byte)
    outRaster.SetGeoTransform((x0, resolution, 0, y0, 0, -resolution))
    for i in range(nb_bands):
        outband = outRaster.GetRasterBand(i+1)
        outband.WriteArray(image[i,:,:])
    outSpatialRef = osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(EPSG)
    outRaster.SetProjection(outSpatialRef.ExportToWkt())


def getOrthoImage(shot, x0, y0, nbCouleurs, path_ortho):
    path = os.path.join(outdir, "Ort_{}.tif".format(shot.nom))
    inputds = gdal.Open(path)
    geotransform = inputds.GetGeoTransform()

    j = (x0-geotransform[0])/resolution
    i = (geotransform[3]-y0)/resolution

    min_c = int(np.floor(j))
    max_c = min_c+tileSize
    min_l = int(np.floor(i))
    max_l = min_l+tileSize

    path2 = os.path.join(path_ortho, "Ort_{}.tif".format(shot.nom))
    inputds = gdal.Open(path2)

    # Coordonnées pour lire une partie de l'image même si dans l'idéal on a besoin d'une partie en-dehors de l'image
    # HistCat (2026-10-15): the read window ends at RasterXSize / RasterYSize (exclusive); with RasterXSize-1 the last
    # column and the last row of every ortho were never read.
    min_c_read = max(0, min_c)
    max_c_read = min(inputds.RasterXSize, max_c)
    min_l_read = max(0, min_l)
    max_l_read = min(inputds.RasterYSize, max_l)

    if max_c_read-min_c_read <= 0 or max_l_read-min_l_read <= 0:
        return None

    image = inputds.ReadAsArray(min_c_read, min_l_read, max_c_read-min_c_read, max_l_read-min_l_read)
    finalImage = np.zeros((nbCouleurs, tileSize, tileSize), dtype=np.uint8)
    finalImage[:,min_l_read-min_l:max_l_read-min_l, min_c_read-min_c:max_c_read-min_c] = image

    return finalImage


def get_path_ortho():
    """
    S'il y a trop de trous dans les orthos, il est possible que la correction radiométrique n'ait pas fonctionné.
    En effet, elle se base sur des points qui se recouvrent entre les images
    Dans ce cas, on utilise les images sans correction radiométrique
    """
    with open(os.path.join(radiom, "ini", "coef_reetal_walis.txt"), "r") as f:
        for line in f:
            if "nan" in line or "Exception" in line:
                logger.warning("Attention, la correction radiométrique a échoué ! L'ortho n'utilisera donc pas la correction radiométrique")
                return outdir
    return os.path.join(radiom, "ini", "corr")


def createOrthoTile(mosaic, shotsFiltered, x0, y0, nbCouleurs, path_ortho, shots):
    """
    HistCat (2026-10-15): each pixel takes the photo of its mosaic polygon, as before; a pixel of the chantier that is
    still empty (its photo has no data there, or no polygon covers it) takes, among the photos that do have data there,
    the one whose camera centre is nearest (least relief displacement). Only photos whose ortho was cut to the photo
    frame lend pixels (remplissage, from HISTCAT_cadre.json): in an uncut ortho, "data" includes the scanner surround.
    Pixels outside the chantier stay empty.
    Before: a tile with a single polygon took that photo everywhere, even outside its polygon and the chantier; a photo
    without data in the tile was replaced by another one over its whole polygon; empty pixels stayed black although
    another photo covered them. Returns (tile or None, pixels filled from another photo, chantier pixels left empty).
    """
    orthoFinale = np.zeros((nbCouleurs, tileSize, tileSize), dtype=np.uint8)
    lues = {}
    for i, shot in enumerate(shotsFiltered):
        if shot.nom not in lues:
            lues[shot.nom] = getOrthoImage(shot, x0, y0, nbCouleurs, path_ortho)
        orthoImage = lues[shot.nom]
        if orthoImage is not None:
            orthoFinale = np.where(mosaic == i + 1, orthoImage, orthoFinale)

    # Pixels of the chantier (pixel centre inside the bbox) still empty
    xs = x0 + (np.arange(tileSize) + 0.5) * resolution
    ys = y0 - (np.arange(tileSize) + 0.5) * resolution
    dans_chantier = ((ys >= bbox[1]) & (ys <= bbox[3]))[:, None] & ((xs >= bbox[0]) & (xs <= bbox[2]))[None, :]
    rows, cols = np.nonzero(dans_chantier & ~orthoFinale.any(axis=0))
    rempli = 0
    if rows.size:
        tuile = (x0, y0 - tileSize * resolution, x0 + tileSize * resolution, y0)
        best = np.full(rows.size, np.inf)
        source = np.full(rows.size, -1)
        candidats = []
        for shot in shots:
            b = ortho_bounds.get(shot.nom)
            if not remplissage.get(shot.nom) or b is None or b[0] >= tuile[2] or b[2] <= tuile[0] or b[1] >= tuile[3] or b[3] <= tuile[1]:
                continue
            if shot.nom not in lues:
                lues[shot.nom] = getOrthoImage(shot, x0, y0, nbCouleurs, path_ortho)
            orthoImage = lues[shot.nom]
            if orthoImage is None:
                continue
            valide = orthoImage[:, rows, cols].any(axis=0)
            d = (xs[cols] - shot.x_pos) ** 2 + (ys[rows] - shot.y_pos) ** 2
            mieux = valide & (d < best)
            best[mieux] = d[mieux]
            source[mieux] = len(candidats)
            candidats.append(orthoImage)
        for k, orthoImage in enumerate(candidats):
            sel = source == k
            orthoFinale[:, rows[sel], cols[sel]] = orthoImage[:, rows[sel], cols[sel]]
        rempli = int((source >= 0).sum())
    vide = int(rows.size) - rempli
    if not orthoFinale.any():
        return None, rempli, vide
    return orthoFinale, rempli, vide


def createOrthoProcess(work_data):
    x0 = work_data[0]
    y0 = work_data[1]
    nbCouleurs = work_data[2]
    EPSG = work_data[3]
    path_ortho = work_data[4]
    shots = work_data[5]

    # On calcule la mosaïque
    mosaic, shots_filtered = computeMosaic(x0, y0, shots)

    # On crée l'ortho pour la tuile
    ortho, rempli, vide = createOrthoTile(mosaic, shots_filtered, x0, y0, nbCouleurs, path_ortho, shots)
    if ortho is not None:
        # On sauvegarde l'ortho
        saveImage(ortho, os.path.join(outdir, "{}_{}_ortho.tif".format(x0, y0)), x0, y0, EPSG)
    return rempli, vide



def createTiles(bbox, shots, nbCouleurs, EPSG, path_ortho):
    work_data = []
    # On crée un tableau numpy qui contient les positions des sommets de prise de vue pour tous les clichés
    pas = int(tileSize*resolution)
    # Pour chaque tuile, on remplit work_data avec les paramètres pour le traitement
    for x0 in range(int(bbox[0]), int(bbox[2]), pas):
        for y0 in range(int(bbox[3]), int(bbox[1]), -pas):
            work_data.append([x0, y0, nbCouleurs, EPSG, path_ortho, shots])
    
    # On parallélise le traitement
    rempli, vide = 0, 0
    with Pool(processes=nb_cpus) as pool:
        with tqdm(total=len(work_data), desc="Calcul de l'ortho") as pbar:
            for r, v in pool.map(createOrthoProcess, work_data):
                rempli += r
                vide += v
                pbar.update()
    km2 = resolution * resolution / 1e6
    surface = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / 1e6
    message = (f"chantier {surface:.1f} km2: {rempli * km2:.3f} km2 filled from the nearest photo with data "
               f"({sum(remplissage.values())} of {len(remplissage)} photos may lend pixels: {raison_remplissage}), "
               f"{vide * km2:.3f} km2 left empty")
    logger.info(message)
    print(f"HISTCAT FILL: {message}")


def get_ortho_bounds(shots):
    """HistCat (2026-10-15): (x_min, y_min, x_max, y_max) of each photo's ortho, None if it has no ortho."""
    bounds = {}
    for shot in shots:
        ds = gdal.Open(os.path.join(outdir, "Ort_{}.tif".format(shot.nom)))
        if ds is None:
            bounds[shot.nom] = None
            continue
        gt = ds.GetGeoTransform()
        bounds[shot.nom] = (gt[0], gt[3] + ds.RasterYSize * gt[5], gt[0] + ds.RasterXSize * gt[1], gt[3])
    return bounds


def get_remplissage(shots):
    """HistCat (2026-10-15): photo -> may lend pixels to the fill, i.e. create_orthos_OIS_Reech.py cut its ortho to the
    photo frame (HISTCAT_cadre.json in the orthos' folder); and why."""
    path = os.path.join(outdir, "HISTCAT_cadre.json")
    if not os.path.isfile(path):
        return {shot.nom: False for shot in shots}, "orthos not cut to the photo frame (no HISTCAT_cadre.json), no fill"
    with open(path) as f:
        kept = json.load(f).get("kept", {})
    ok = {shot.nom: kept.get(shot.nom) is not None for shot in shots}
    return ok, f"orthos cut to the photo frame for {sum(ok.values())} photos"

def read_mosaique(shots, mosaic_file):
    mosaique = []
    emprises_mosaique = gpd.read_file(mosaic_file)
    for i in range(emprises_mosaique.shape[0]):
        raw = emprises_mosaique.iloc[i]
        geometry = raw.geometry
        for j in range(len(shots)):
            if shots[j].imagePath==raw["shot"]:
                mosaique.append((geometry, j))
    return mosaique


os.makedirs(outdir, exist_ok=True)

mnt = MNT(mnt_path)

# On charge la boite englobante du chantier
bbox = load_bbox("metadata")

resolution = getResolution()
nbCouleurs = getNbCouleurs("metadata")
EPSG = getEPSG("metadata")

# On crée un objet shot par image
shots = read_ori(ori_path, ta_path, EPSG)
mosaique = read_mosaique(shots, mosaic_file)

# On récupère le réeprtoire où se trouvent les orthos pour chaque image
# Si la correction radiométrique n'a pas fonctionné, alors on prend celles sans correction radiométrique
path_ortho = get_path_ortho()
ortho_bounds = get_ortho_bounds(shots)
remplissage, raison_remplissage = get_remplissage(shots)

# Crée les tuiles d'ortho
createTiles(bbox, shots, nbCouleurs, EPSG, path_ortho)