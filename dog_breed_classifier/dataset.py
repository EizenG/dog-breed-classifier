"""Téléchargement et organisation du Stanford Dogs Dataset."""
import hashlib
import shutil
import tarfile
import urllib.request
from pathlib import Path

import yaml
from loguru import logger

from dog_breed_classifier.config import PROJ_ROOT, RAW_DATA_DIR

DATASET_URL = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
ARCHIVE_NAME = "images.tar"


def download_dataset(dest: Path) -> Path:
    """Télécharge l'archive du Stanford Dogs Dataset.

    Args:
        dest: Dossier de destination pour l'archive.

    Returns:
        Chemin vers l'archive téléchargée.
    """
    dest.mkdir(parents=True, exist_ok=True)
    archive_path = dest / ARCHIVE_NAME

    if archive_path.exists():
        logger.info("Archive déjà présente, téléchargement ignoré.")
        return archive_path

    logger.info(f"Téléchargement depuis {DATASET_URL} ...")
    urllib.request.urlretrieve(DATASET_URL, archive_path)
    logger.info("Téléchargement terminé.")
    return archive_path


def extract_dataset(archive_path: Path, dest: Path) -> Path:
    """Extrait l'archive dans le dossier de destination.

    Args:
        archive_path: Chemin vers l'archive .tar.
        dest: Dossier d'extraction.

    Returns:
        Chemin vers le dossier Images extrait.

    Raises:
        FileNotFoundError: Si l'archive n'existe pas.
    """
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive introuvable : {archive_path}")

    images_path = dest / "Images"
    if images_path.exists():
        logger.info("Dataset déjà extrait, extraction ignorée.")
        return images_path

    logger.info("Extraction de l'archive ...")
    with tarfile.open(archive_path) as tar:
        tar.extractall(dest)
    logger.info("Extraction terminée.")
    return images_path


def organize_by_breed(images_path: Path, output_path: Path) -> None:
    """Réorganise les images par nom de race dans output_path.

    Les dossiers Stanford ont le format 'n02085620-Chihuahua'.
    Cette fonction renomme chaque dossier en nom de race uniquement.

    Args:
        images_path: Dossier contenant les sous-dossiers par race (format ImageNet).
        output_path: Dossier de sortie avec un sous-dossier par race.
    """
    if output_path.exists():
        logger.info("Dossier breeds déjà présent, organisation ignorée.")
        return

    output_path.mkdir(parents=True, exist_ok=True)
    breed_dirs = sorted([d for d in images_path.iterdir() if d.is_dir()])
    logger.info(f"{len(breed_dirs)} races trouvées dans le dataset.")

    for breed_dir in breed_dirs:
        breed_name = breed_dir.name.split("-", 1)[-1]
        breed_output = output_path / breed_name
        breed_output.mkdir(exist_ok=True)

        images = list(breed_dir.glob("*.jpg"))
        for img in images:
            shutil.copy2(img, breed_output / img.name)

        logger.info(f"  {breed_name} : {len(images)} images")


def filter_breeds(breeds_path: Path, output_path: Path, params: dict) -> None:
    """Copie uniquement les races sélectionnées dans output_path.

    Args:
        breeds_path: Dossier contenant toutes les races organisées.
        output_path: Dossier de sortie avec les 10 races sélectionnées.
        params: Paramètres du projet (liste des races).
    """
    if output_path.exists():
        logger.info("Dossier selected déjà présent, filtrage ignoré.")
        return

    races = params["data"]["races"]
    output_path.mkdir(parents=True, exist_ok=True)

    for race in races:
        race_src = breeds_path / race
        if not race_src.exists():
            logger.warning(f"Race introuvable : {race}, ignorée.")
            continue

        race_dest = output_path / race
        shutil.copytree(race_src, race_dest)
        logger.info(f"  {race} : {len(list(race_dest.glob('*.jpg')))} images")


def remove_duplicates(root: Path) -> int:
    """Supprime les images dupliquées (même contenu, hash MD5 identique) sous root.

    Parcourt récursivement tous les .jpg. Pour chaque hash déjà vu, supprime
    le fichier en double et conserve la première occurrence rencontrée.

    Args:
        root: Dossier racine à analyser (ex : data/raw/selected).

    Returns:
        Nombre de fichiers supprimés.
    """
    seen: dict[str, Path] = {}
    removed = 0

    for img_path in sorted(root.rglob("*.jpg")):
        digest = hashlib.md5(img_path.read_bytes()).hexdigest()
        if digest in seen:
            logger.info(f"Doublon supprimé : {img_path} (identique à {seen[digest]})")
            img_path.unlink()
            removed += 1
        else:
            seen[digest] = img_path

    logger.info(f"remove_duplicates : {removed} doublon(s) supprimé(s) sur {len(seen) + removed} images.")
    return removed


def main() -> None:
    """Téléchargement, organisation et filtrage des races sélectionnées."""
    params_path = PROJ_ROOT / "params.yaml"
    with open(params_path) as f:
        params = yaml.safe_load(f)

    breeds_path = RAW_DATA_DIR / "breeds"

    if not breeds_path.exists():
        archive = download_dataset(RAW_DATA_DIR)
        images_path = extract_dataset(archive, RAW_DATA_DIR)
        organize_by_breed(images_path, breeds_path)
    else:
        logger.info("Dossier breeds déjà présent, téléchargement et extraction ignorés.")

    filter_breeds(breeds_path, RAW_DATA_DIR / "selected", params)
    logger.info(f"Données filtrées disponibles dans {RAW_DATA_DIR / 'selected'}")


if __name__ == "__main__":
    main()
