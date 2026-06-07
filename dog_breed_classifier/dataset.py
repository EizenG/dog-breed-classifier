"""Téléchargement et organisation du Stanford Dogs Dataset."""
import shutil
import tarfile
import urllib.request
from pathlib import Path

from loguru import logger

from dog_breed_classifier.config import RAW_DATA_DIR

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


def main() -> None:
    """Pipeline de téléchargement et d'organisation des données brutes."""
    archive = download_dataset(RAW_DATA_DIR)
    images_path = extract_dataset(archive, RAW_DATA_DIR)
    organize_by_breed(images_path, RAW_DATA_DIR / "breeds")
    logger.info(f"Données disponibles dans {RAW_DATA_DIR / 'breeds'}")


if __name__ == "__main__":
    main()
