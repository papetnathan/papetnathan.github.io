import os
import re
import json
import time
from pathlib import Path
from openai import OpenAI

# ── Config ─────────────────────────────────────────────────────────────────────
KNOWLEDGE_FILE  = Path(__file__).parent.parent / "knowledge" / "nathan.md"
OUTPUT_FILE     = Path(__file__).parent.parent / "knowledge" / "embeddings.json"
EMBEDDING_MODEL = "text-embedding-3-small"   # 1536 dims, pas cher, très bon
# ───────────────────────────────────────────────────────────────────────────────


def parse_chunks(md_path: Path) -> list[dict]:
    """Parse le fichier markdown en chunks structurés.
    
    Format attendu :
        # CHUNK: nom_du_chunk
        Contenu du chunk...
    """
    text = md_path.read_text(encoding="utf-8")
    raw_chunks = re.split(r"^# CHUNK:", text, flags=re.MULTILINE)

    chunks = []
    for raw in raw_chunks:
        raw = raw.strip()
        if not raw:
            continue
        lines = raw.split("\n", 1)
        chunk_id = lines[0].strip()
        content  = lines[1].strip() if len(lines) > 1 else ""
        if content:
            chunks.append({"id": chunk_id, "text": content})

    return chunks


def embed_chunks(chunks: list[dict], client: OpenAI) -> list[dict]:
    """Appelle l'API OpenAI pour embedder tous les chunks."""
    texts = [c["text"] for c in chunks]

    print(f"📡 Embedding {len(texts)} chunks avec {EMBEDDING_MODEL}...")
    
    # On envoie tout en une seule requête batch (plus efficace)
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    for i, chunk in enumerate(chunks):
        chunk["embedding"] = response.data[i].embedding

    print(f"✅ {len(chunks)} embeddings générés ({len(chunks[0]['embedding'])} dimensions)")
    return chunks


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("❌ OPENAI_API_KEY non définie. Fais : export OPENAI_API_KEY='sk-...'")

    client = OpenAI(api_key=api_key)

    # 1. Parser la knowledge base
    print(f"📖 Lecture de {KNOWLEDGE_FILE}...")
    chunks = parse_chunks(KNOWLEDGE_FILE)
    print(f"   → {len(chunks)} chunks trouvés : {[c['id'] for c in chunks]}")

    # 2. Générer les embeddings
    chunks_with_embeddings = embed_chunks(chunks, client)

    # 3. Sauvegarder
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "model":  EMBEDDING_MODEL,
        "chunks": chunks_with_embeddings,
    }
    OUTPUT_FILE.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    size_kb = OUTPUT_FILE.stat().st_size / 1024
    print(f"\n💾 Sauvegardé dans {OUTPUT_FILE} ({size_kb:.1f} KB)")
    print("🚀 Tu peux maintenant commiter knowledge/embeddings.json dans ton repo !")


if __name__ == "__main__":
    main()
