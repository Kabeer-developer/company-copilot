from typing import BinaryIO

from pypdf import PdfReader

from app.extractors import ExtractionError


def extract_pdf(stream: BinaryIO) -> str:
    try:
        stream.seek(0)
        reader = PdfReader(stream)
        if reader.is_encrypted:
            raise ExtractionError("PDF is encrypted and cannot be processed.")
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    except ExtractionError:
        raise
    except Exception as exc:
        raise ExtractionError("PDF text extraction failed. Check that the PDF is valid and readable.") from exc
