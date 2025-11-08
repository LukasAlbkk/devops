import os, time, threading, json, pickle
from flask import Flask, request, jsonify

MODEL_PATH   = os.getenv("MODEL_PATH", "/shared/model/rules_model.pkl")
CODE_VERSION = os.getenv("CODE_VERSION", "0.3.0")
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
    if not input_norm:
        return []

    rules = app.model["rules"]

    # score por consequente
    scores = {}

    # para cada antecedente que é subconjunto das músicas do usuário
    input_set = set(input_norm)

    # Contador de regras aplicadas (para debug)
    rules_applied = 0

    # DEBUG: Log input songs normalized
    app.logger.info(f"[API DEBUG] Input normalized: {input_norm}")
    app.logger.info(f"[API DEBUG] Total antecedents in model: {len(rules)}")

    # DEBUG: Sample first 5 antecedents to see format
    sample_ants = list(rules.keys())[:5]
    app.logger.info(f"[API DEBUG] Sample antecedents: {sample_ants}")

    # mapeia todas as combinações possíveis presentes nos antecedentes
    for ant_t, outs in rules.items():
        # As músicas já estão normalizadas no modelo
        ant_set = set(ant_t)
        if ant_set.issubset(input_set) and len(ant_set) > 0:
            rules_applied += 1
            for cons_t, conf in outs:
                for c in cons_t:
                    # c já está normalizado no modelo
                    if c in input_set:  # não recomendar o que já veio
                        continue
                    scores[c] = scores.get(c, 0.0) + float(conf)

    app.logger.info(f"[API] Regras aplicadas: {rules_applied}, Candidatos únicos: {len(scores)}")

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

    # Validação de entrada
    if not isinstance(songs, list):
        return jsonify({"error": "songs deve ser uma lista de strings"}), 400

    if len(songs) == 0:
        return jsonify({
            "error": "Lista de músicas vazia. Forneça pelo menos 1 música.",
            "example": {"songs": ["Song Name 1", "Song Name 2"]}
        }), 400

    # Validar se todos os itens são strings
    invalid_items = [i for i, s in enumerate(songs) if not isinstance(s, str) or not str(s).strip()]
    if invalid_items:
        return jsonify({
            "error": f"Itens inválidos nas posições: {invalid_items}. Todos devem ser strings não vazias."
        }), 400

    # Log da requisição
    app.logger.info(f"[API] Recomendando para {len(songs)} música(s)")

    # Verifica se modelo existe
    if not app.model:
        app.logger.warning("[API] Modelo não carregado. Retornando lista vazia.")
        return jsonify({
            "songs": [],
            "version": CODE_VERSION,
            "model_date": None,
            "warning": "Modelo não carregado ainda. Aguarde o treinamento completar."
        }), 200

    if "rules" not in app.model or len(app.model.get("rules", {})) == 0:
        app.logger.warning("[API] Modelo carregado mas sem regras.")
        return jsonify({
            "songs": [],
            "version": CODE_VERSION,
            "model_date": app.model.get("model_date"),
            "warning": "Modelo sem regras de associação. Talvez MIN_SUP esteja muito alto."
        }), 200

    recs = _recommend(songs)
    model_date = app.model.get("model_date", None)

    app.logger.info(f"[API] Retornando {len(recs)} recomendações")

    return jsonify({
        "songs": recs,
        "version": CODE_VERSION,
        "model_date": model_date,
        "input_songs_count": len(songs),
        "total_rules": len(app.model.get("rules", {}))
    })

if __name__ == "__main__":
    # inicia watcher em background
    t = threading.Thread(target=_watch_model, daemon=True)
    t.start()
    # tenta carregar uma vez
    _try_load_model()
    app.run(host=HOST, port=PORT)
