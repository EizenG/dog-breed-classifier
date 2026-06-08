"""Split stratifié du dataset et organisation pour l'entraînement."""
import shutil
from pathlib import Path

import yaml
from loguru import logger
from sklearn.model_selection import train_test_split

from dog_breed_classifier.config import PROCESSED_DATA_DIR, PROJ_ROOT, RAW_DATA_DIR


def load_image_paths(selected_path: Path, races: list[str]) -> tuple[list[Path], list[str]]:
    """Collecte les chemins d'images et leurs labels depuis selected_path."""
    images: list[Path] = []
    labels: list[str] = []

    for race in races:
        race_dir = selected_path / race
        if not race_dir.exists():
            logger.warning(f"Race introuvable : {race}, ignorée.")
            continue
        imgs = sorted(race_dir.glob("*.jpg"))
        images.extend(imgs)
        labels.extend([race] * len(imgs))
        logger.info(f"  {race} : {len(imgs)} images")

    return images, labels


def split_and_copy(
    images: list[Path],
    labels: list[str],
    output_path: Path,
    train_ratio: float,
    val_ratio: float,
) -> None:
    """Divise les images en train/val/test et les copie dans output_path."""
    test_ratio = round(1.0 - train_ratio - val_ratio, 10)

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        images, labels, test_size=(1.0 - train_ratio), stratify=labels, random_state=42
    )

    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=(1.0 - val_ratio_adjusted), stratify=y_tmp, random_state=42
    )

    splits = {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }

    for split_name, (split_images, split_labels) in splits.items():
        logger.info(f"  {split_name} : {len(split_images)} images")
        for img_path, label in zip(split_images, split_labels):
            dest_dir = output_path / split_name / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, dest_dir / img_path.name)


def main() -> None:
    """Split stratifié et organisation des images dans data/processed/."""
    if (PROCESSED_DATA_DIR / "train").exists():
        logger.info("Split déjà effectué, ignoré.")
        return

    params_path = PROJ_ROOT / "params.yaml"
    with open(params_path) as f:
        params = yaml.safe_load(f)

    races = params["data"]["races"]
    train_ratio = params["data"]["split"]["train"]
    val_ratio = params["data"]["split"]["val"]

    selected_path = RAW_DATA_DIR / "selected"
    logger.info("Chargement des chemins d'images...")
    images, labels = load_image_paths(selected_path, races)
    logger.info(f"Total : {len(images)} images — {len(set(labels))} races")

    logger.info("Split et copie des images...")
    split_and_copy(images, labels, PROCESSED_DATA_DIR, train_ratio, val_ratio)
    logger.success(f"Données préparées dans {PROCESSED_DATA_DIR}")


if __name__ == "__main__":
    main()
