from typing import BinaryIO

from docx import Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.extractors import ExtractionError


def extract_docx(stream: BinaryIO) -> str:
    try:
        stream.seek(0)
        document = Document(stream)
        blocks: list[str] = []
        for child in document.element.body.iterchildren():
            if isinstance(child, CT_P):
                text = Paragraph(child, document).text
                if text.strip():
                    blocks.append(text)
            elif isinstance(child, CT_Tbl):
                table = Table(child, document)
                for row in table.rows:
                    cells = [cell.text.strip() for cell in row.cells]
                    if any(cells):
                        blocks.append(" | ".join(cells))
        return "\n".join(blocks)
    except Exception as exc:
        raise ExtractionError("DOCX text extraction failed. Check that the file is a valid Word document.") from exc
