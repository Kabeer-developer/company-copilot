from typing import BinaryIO

from app.extractors import ExtractionError


def extract_txt(stream: BinaryIO) -> str:
    try:
        stream.seek(0)
        contents = stream.read()
        try:
            return contents.decode("utf-8-sig")
        except UnicodeDecodeError:
            if b"\x00" in contents:
                raise ExtractionError("TXT decoding failed. Use UTF-8 or Windows-1252 text encoding.")
            return contents.decode("cp1252")
    except ExtractionError:
        raise
    except Exception as exc:
        raise ExtractionError("TXT text extraction failed.") from exc
