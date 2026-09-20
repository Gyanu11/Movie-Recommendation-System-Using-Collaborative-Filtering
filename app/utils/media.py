"""Profile picture uploads."""
from __future__ import annotations

import secrets
from pathlib import Path

from PIL import Image, UnidentifiedImageError

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def save_profile_picture(file_storage, upload_folder: str | Path, size=(125, 125)) -> str:
    """Store an uploaded picture as a thumbnail and return its filename.

    The client-supplied filename is never trusted: only its extension is kept
    (and only if allow-listed), and the stored name is random. That removes any
    path-traversal or overwrite risk from a hostile upload.

    Raises:
        ValueError: the file is not a readable JPEG or PNG image.
    """
    upload_folder = Path(upload_folder)
    upload_folder.mkdir(parents=True, exist_ok=True)

    extension = Path(file_storage.filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError("Profile pictures must be JPG or PNG files.")

    filename = f"{secrets.token_hex(8)}{extension}"
    destination = upload_folder / filename

    try:
        with Image.open(file_storage) as image:
            image.thumbnail(size)
            # Pillow needs RGB to write a JPEG; palette and alpha modes fail.
            if extension in {".jpg", ".jpeg"} and image.mode not in {"RGB", "L"}:
                image = image.convert("RGB")
            image.save(destination)
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("That file could not be read as an image.") from exc

    return filename
