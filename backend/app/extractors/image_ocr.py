import shutil
from typing import BinaryIO

from PIL import Image, UnidentifiedImageError
import pytesseract
from pytesseract import TesseractNotFoundError

from app.extractors import ExtractionError


def extract_image_text(stream: BinaryIO) -> str:
    if shutil.which(pytesseract.pytesseract.tesseract_cmd) is None:
        raise ExtractionError("OCR engine unavailable. Install Tesseract OCR and make tesseract.exe available on PATH.")
    try:
        stream.seek(0)
        with Image.open(stream) as image:
            image.load()
            return pytesseract.image_to_string(image)
    except TesseractNotFoundError as exc:
        raise ExtractionError("OCR engine unavailable. Install Tesseract OCR and make tesseract.exe available on PATH.") from exc
    except UnidentifiedImageError as exc:
        raise ExtractionError("Image OCR failed because the image file is invalid.") from exc
    except Exception as exc:
        raise ExtractionError("Image OCR failed. Check the image and Tesseract language data.") from exc
