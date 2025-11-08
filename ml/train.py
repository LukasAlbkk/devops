# /home/lucascosta/devops/ml/train.py (NOVO CÓDIGO)

import os, sys, io, json, time, pickle, tempfile, random
from urllib.parse import urlparse
import requests
import pandas as pd
from fpgrowth_py import fpgrowth

# === Config por ENV ===
DATASET_URL  = os.getenv("DATASET_URL")
PLAYLIST_COL = os.getenv("PLAYLIST_COL", "pid")
SONG_COL     = os.getenv("SONG_COL", "track_name")
MIN_SUP      = float(os.getenv("MIN_SUP", "0.001"))
MIN_CONF     = float(os.getenv("MIN_CONF", "0.2"))
MODEL_DIR    = os.getenv("MODEL_DIR", "/shared/model")
MODEL_NAME   = os.getenv("MODEL_NAME", "rules_model.pkl")

# NOVA VARIÁVEL: Limita o número de baskets para controlar o uso de memória
# 50k é um bom valor para ficar abaixo de 2G de RAM com MIN_SUP baixo
MAX_BASKETS  = int(os.getenv("MAX_BASKETS", "50000")) 

assert DATASET_URL, "Defina DATASET_URL"

os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
META_PATH  = os.path.join(MODEL_DIR, "model_meta.json")

def _download_or_open(url: str) -> io.BytesIO:
    p = urlparse(url)
    if p.scheme in ("http", "https"):
        r = requests.get(url, timeout=600)
        r.raise_for_status()
        return io.BytesIO(r.content)
    if p.scheme == "file":
        path = p.path
        with open(path, "rb") as f:
            return io.BytesIO(f.read())
    # caminho local cru
    if os.path.exists(url):
        with open(url, "rb") as f:
            return io.BytesIO(f.read())
    raise ValueError(f"Não consegui abrir: {url}")

def _infer_baskets(df: pd.DataFrame) -> list[list[str]]:
    cols = set(c.lower() for c in df.columns)
    # normaliza nomes
    def pick(name_candidates):
        for n in name_candidates:
            for c in df.columns:
                if c.lower() == n.lower():
                    return c
        return None

    pl_col = pick([PLAYLIST_COL, "playlist_id", "pid", "playlist"])
    sg_col = pick([SONG_COL, "song_name", "track_name", "track", "track_uri", "song"])

    if pl_col and sg_col:
        # formato row-wise: (playlist_id, song)
        grouped = df.groupby(pl_col)[sg_col].apply(lambda s: [str(x).strip() for x in s.dropna().astype(str).tolist()])
        baskets = [list(dict.fromkeys(x)) for x in grouped.tolist()]  # remove duplicados preservando ordem
        return [b for b in baskets if len(b) >= 2]

    # formato com coluna de lista/JSON em 'tracks'/'songs'
    list_col = pick(["tracks", "songs", "itens", "items"])
    if list_col:
        baskets = []
        for raw in df[list_col].dropna().astype(str):
            try:
                # tenta JSON
                v = json.loads(raw)
                if isinstance(v, list):
                    basket = [str(x).strip() for x in v if str(x).strip()]
                    if len(basket) >= 2:
                        baskets.append(list(dict.fromkeys(basket)))
                    continue
            except Exception:
                pass
            # fallback: separadores comuns
            for sep in ["|", ";", "~~", ","]:
                if sep in raw:
                    parts = [p.strip() for p in raw.split(sep)]
                    parts = [p for p in parts if p]
                    if len(parts) >= 2:
                        baskets.append(list(dict.fromkeys(parts)))
                    break
        return baskets

    raise ValueError("Não consegui identificar colunas de playlist e música. Ajuste PLAYLIST_COL/SONG_COL.")

def train_rules(baskets: list[list[str]]):
    # fpgrowth_py retorna tuples de sets
    freq, rules = fpgrowth(baskets, minSupRatio=MIN_SUP, minConf=MIN_CONF)
    # normaliza em dict: antecedente(tuple)-> list[(consequente(tuple), conf)]
    rules_map = {}
    for ant, cons, conf in rules:
        ant_t = tuple(sorted([str(x) for x in ant]))
        cons_t = tuple(sorted([str(x) for x in cons]))
        rules_map.setdefault(ant_t, []).append((cons_t, float(conf)))
    return rules_map

def save_model(rules_map: dict):
    model = {
        "kind": "fp_rules",
        "rules": rules_map,
        "min_sup": MIN_SUP,
        "min_conf": MIN_CONF,
        "dataset_url": DATASET_URL,
        "model_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "song_norm": "casefold"
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(META_PATH, "w") as f:
        json.dump({k: (v if k != "rules" else f"{len(rules_map)} antecedents") for k, v in model.items()}, f, ensure_ascii=False, indent=2)

def main():
    buf = _download_or_open(DATASET_URL)
    df = pd.read_csv(buf, dtype=str, keep_default_na=False)
    
    baskets = _infer_baskets(df)
    print(f"[ML] Baskets totais encontrados: {len(baskets)}")

    # --- LÓGICA DE AMOSTRAGEM ---
    # Se tivermos mais baskets do que o limite, fazemos uma amostra aleatória
    if len(baskets) > MAX_BASKETS:
        print(f"[ML] Dataset muito grande. Amostrando {MAX_BASKETS} baskets aleatoriamente...")
        baskets = random.sample(baskets, MAX_BASKETS)
    # --- FIM DA LÓGICA DE AMOSTRAGEM ---

    print(f"[ML] Baskets a usar no treino: {len(baskets)}")
    
    rules = train_rules(baskets)
    print(f"[ML] Antecedentes gerados: {len(rules)}")
    
    save_model(rules)
    print(f"[ML] Modelo salvo em: {MODEL_PATH}")

if __name__ == "__main__":
    main()