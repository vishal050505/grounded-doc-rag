def chunk_text(text, min_chunk_size=200, max_chunk_size=600, overlap=100):
    """
    Structure-aware semantic chunking for RAG.
    Works strictly on document text.
    """

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        # If a single paragraph is very large, split it
        if len(para) > max_chunk_size:
            words = para.split()
            temp = ""

            for word in words:
                temp += word + " "
                if len(temp) >= max_chunk_size:
                    chunks.append(temp.strip())
                    temp = temp[-overlap:] if overlap < len(temp) else temp

            if temp.strip():
                chunks.append(temp.strip())
            continue

        # Merge paragraphs into a chunk
        if len(current_chunk) + len(para) <= max_chunk_size:
            current_chunk += " " + para
        else:
            if len(current_chunk) >= min_chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = current_chunk[-overlap:] + " " + para
            else:
                current_chunk += " " + para

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks
