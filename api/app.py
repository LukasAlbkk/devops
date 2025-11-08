import os, time, threading, json, pickle
from flask import Flask, request, jsonify

MODEL_PATH   = os.getenv("MODEL_PATH", "/shared/model/rules_model.pkl")
CODE_VERSION = os.getenv("CODE_VERSION", "0.1.0")
PORT         = int(os.getenv("PORT", "5000"))
HOST         = "0.0.0.0"

app = Flask(__name__)
app.model = None
app.model_mtime = None

def _load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def _try_load_model():
    try:
        mtime = os.path.getmtime(MODEL_PATH)
        if app.model is None or app.model_mtime != mtime:
            app.model = _load_model()
            app.model_mtime = mtime
            app.logger.info(f"[API] Modelo carregado/atualizado ({time.ctime(mtime)})")
    except FileNotFoundError:
        app.logger.warning(f"[API] Modelo não encontrado em {MODEL_PATH}")

def _watch_model(loop_sec=5):
    while True:
        try:
            _try_load_model()
        except Exception as e:
            app.logger.error(f"[API] Erro ao recarregar modelo: {e}")
        time.sleep(loop_sec)

def _norm(song: str) -> str:
    return str(song).casefold().strip()

def _recommend(input_songs: list[str], top_k: int = 20):
    if not app.model or "rules" not in app.model:
        return []

    input_norm = [_norm(s) for s in input_songs if str(s).strip()]
    rules = app.model["rules"]

    # score por consequente
    scores = {}

    # para cada antecedente que é subconjunto das músicas do usuário
    input_set = set(input_norm)

    # mapeia todas as combinações possíveis presentes nos antecedentes
    for ant_t, outs in rules.items():
        ant_set = set(_norm(x) for x in ant_t)
        if ant_set.issubset(input_set) and len(ant_set) > 0:
            for cons_t, conf in outs:
                for c in cons_t:
                    c_norm = _norm(c)
                    if c_norm in input_set:  # não recomendar o que já veio
                        continue
                    scores[c_norm] = scores.get(c_norm, 0.0) + float(conf)

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [name for name, _ in ranked[:top_k]]

@app.route("/health")
def health():
    return "ok", 200

@app.route("/api/recommend", methods=["POST"])
def recommend():
    # força JSON
    payload = request.get_json(force=True, silent=True) or {}
    songs = payload.get("songs", [])
    if not isinstance(songs, list):
        return jsonify({"error": "songs deve ser uma lista de strings"}), 400

    recs = _recommend(songs)
    model_date = (app.model or {}).get("model_date", None)
    return jsonify({
        "songs": recs,
        "version": CODE_VERSION,
        "model_date": model_date
    })

if __name__ == "__main__":
    # inicia watcher em background
    t = threading.Thread(target=_watch_model, daemon=True)
    t.start()
    # tenta carregar uma vez
    _try_load_model()
    app.run(host=HOST, port=PORT)
