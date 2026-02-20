"""
api/chat.py  —  Vercel Serverless Function (Python)
────────────────────────────────────────────────────
Pipeline RAG :
  1. Embed la question de l'utilisateur (OpenAI text-embedding-3-small)
  2. Cosine similarity avec les chunks pré-calculés (numpy, chargés en mémoire)
  3. Récupère les top-k chunks les plus pertinents
  4. Construit un prompt augmenté et appelle gpt-4o-mini
"""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI

# ── Config ─────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL  = "text-embedding-3-small"
CHAT_MODEL       = "gpt-4o-mini"
TOP_K            = 3          # nombre de chunks récupérés
MAX_TOKENS       = 350
TEMPERATURE      = 0.65
MAX_HISTORY      = 6          # messages max gardés en contexte
# ───────────────────────────────────────────────────────────────────────────────

# Chemin vers les embeddings pré-calculés
# En serverless Vercel, __file__ pointe vers api/chat.py
_BASE_DIR        = Path(__file__).parent.parent
EMBEDDINGS_PATH  = _BASE_DIR / "knowledge" / "embeddings.json"


# ── Chargement des embeddings (une fois au cold start) ──────────────────────
def _load_embeddings() -> tuple[list[str], np.ndarray]:
    """Charge le fichier embeddings.json et retourne (textes, matrice numpy)."""
    data = json.loads(EMBEDDINGS_PATH.read_text(encoding="utf-8"))
    chunks   = data["chunks"]
    texts    = [c["text"] for c in chunks]
    matrix   = np.array([c["embedding"] for c in chunks], dtype=np.float32)
    # Normalise les vecteurs ligne par ligne (pour cosine similarity = dot product)
    norms    = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix   = matrix / np.clip(norms, 1e-10, None)
    return texts, matrix


# Chargement au démarrage du worker (mis en cache entre les requêtes tièdes)
try:
    CHUNK_TEXTS, CHUNK_MATRIX = _load_embeddings()
    print(f"[RAG] {len(CHUNK_TEXTS)} chunks chargés ({CHUNK_MATRIX.shape[1]} dims)", file=sys.stderr)
except Exception as e:
    print(f"[RAG] ERREUR chargement embeddings : {e}", file=sys.stderr)
    CHUNK_TEXTS, CHUNK_MATRIX = [], np.array([])


# ── Fonctions RAG ────────────────────────────────────────────────────────────

def embed_query(query: str, client: OpenAI) -> np.ndarray:
    """Retourne le vecteur normalisé de la query."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    vec = np.array(response.data[0].embedding, dtype=np.float32)
    return vec / np.linalg.norm(vec)


def retrieve(query_vec: np.ndarray, top_k: int = TOP_K) -> list[str]:
    """Cosine similarity → top-k chunks."""
    if CHUNK_MATRIX.size == 0:
        return []
    scores  = CHUNK_MATRIX @ query_vec          # dot product = cosine (vecteurs normalisés)
    indices = np.argsort(scores)[::-1][:top_k]
    return [CHUNK_TEXTS[i] for i in indices]


def build_system_prompt(context_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    return f"""Tu es l'assistant personnel de Nathan Papet, un Data Scientist basé à Bordeaux, France.
Tu réponds aux questions des visiteurs de son portfolio de manière chaleureuse, professionnelle et concise.
Réponds toujours en français sauf si le visiteur écrit en anglais.
Tu peux parler à la première personne comme si tu étais Nathan ("Je travaille chez...").

Voici les informations pertinentes issues de la base de connaissance de Nathan :

{context}

RÈGLES :
- Réponds UNIQUEMENT à partir des informations ci-dessus.
- Si la réponse n'est pas dans le contexte, dis-le honnêtement et invite à contacter Nathan par email : nathan.papet@icloud.com
- Ne jamais inventer de diplômes, salaires ou informations absentes du contexte.
- Sois concis (3-4 phrases max) sauf si une explication détaillée est demandée.
- N'hésites pas à donner des petites anecdotes ou détails personnels pour rendre les réponses plus humaines et engageantes.
"""


# ── Handler Vercel ───────────────────────────────────────────────────────────

class handler(BaseHTTPRequestHandler):

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self) -> None:
        try:
            length   = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(length)
            body     = json.loads(raw_body)
        except Exception:
            self._send_json(400, {"error": "Invalid JSON body"})
            return

        messages: list[dict] = body.get("messages", [])
        if not messages or not isinstance(messages, list):
            self._send_json(400, {"error": "messages requis"})
            return

        # Dernière question de l'utilisateur
        last_user_msg = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"),
            None,
        )
        if not last_user_msg:
            self._send_json(400, {"error": "Aucun message utilisateur trouvé"})
            return

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            self._send_json(500, {"error": "OPENAI_API_KEY manquante"})
            return

        client = OpenAI(api_key=api_key)

        try:
            # ── 1. Embed la question ──────────────────────────────────────────
            query_vec = embed_query(last_user_msg, client)

            # ── 2. Retrieval ─────────────────────────────────────────────────
            top_chunks = retrieve(query_vec, top_k=TOP_K)

            # ── 3. Augmented prompt ──────────────────────────────────────────
            system_prompt = build_system_prompt(top_chunks)

            # ── 4. Appel LLM ─────────────────────────────────────────────────
            recent_history = messages[-MAX_HISTORY:]
            response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    *recent_history,
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )

            reply = response.choices[0].message.content
            self._send_json(200, {"reply": reply})

        except Exception as e:
            print(f"[RAG] Erreur pipeline : {e}", file=sys.stderr)
            self._send_json(500, {"error": "Erreur interne du pipeline RAG"})

    # Silence les logs HTTP de BaseHTTPRequestHandler en prod
    def log_message(self, format: str, *args: Any) -> None:
        print(f"[{self.address_string()}] {format % args}", file=sys.stderr)