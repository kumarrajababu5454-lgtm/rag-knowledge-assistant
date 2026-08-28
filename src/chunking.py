import re


def chunk_text(text):
    """
    Split a document into paragraph-based chunks.
    Each paragraph becomes one chunk.
    """

    paragraphs = re.split(r"\n\s*\n", text.strip())

    chunks = []

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if paragraph:
            chunks.append(paragraph)

    return chunks