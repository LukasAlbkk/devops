
import os, io, json, time, pickle, sqlite3, math
from urllib.parse import urlparse
import pandas as pd

DATASET_URL = os.getenv("DATASET_URL")                      
PLAYLIST_COL = os.getenv("PLAYLIST_COL", "pid")
SONG_COL     = os.getenv("SONG_COL", "track_name")
MIN_SUP      = float(os.getenv("MIN_SUP", "0.01"))         
MIN_CONF     = float(os.getenv("MIN_CONF", "0.2"))
MODEL_DIR    = os.getenv("MODEL_DIR", "/shared/model")
MODEL_NAME   = os.getenv("MODEL_NAME", "rules_model.pkl")
CHUNK_ROWS   = int(os.getenv("CHUNK_ROWS", "250000"))      
TMP_DIR      = os.getenv("TMP_DIR", "/tmp")
MAX_RULES_PER_ANT = int(os.getenv("MAX_RULES_PER_ANT", "30"))

assert DATASET_URL, "Defina DATASET_URL"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
META_PATH  = os.path.join(MODEL_DIR, "model_meta.json")

_fim = None
try:
    import fim as _fim   
except Exception:
    try:
        import pyfim as _fim  
    except Exception:
        _fim = None

def _download_or_open(url: str) -> io.BytesIO:
    p = urlparse(url)
    if p.scheme in ("http", "https"):
        import requests
        r = requests.get(url, timeout=600)
        r.raise_for_status()
        return io.BytesIO(r.content)
    if p.scheme == "file":
        path = p.path
        with open(path, "rb") as f:
            return io.BytesIO(f.read())
    if os.path.exists(url):  
        with open(url, "rb") as f:
            return io.BytesIO(f.read())
    raise ValueError(f"Não consegui abrir: {url}")

def _norm(song: str) -> str:
    """Normaliza nome de música removendo sufixos comuns e padronizando (mesma função usada na API)."""
    s = str(song).casefold().strip()

    suffixes = [
        ' - remastered 2011',
        ' - remastered 2009',
        ' - remastered 2010',
        ' - remastered',
        ' - live',
        ' - radio edit',
        ' - album version',
        ' - single version',
        ' - explicit',
        ' - clean',
    ]

    for suffix in suffixes:
        if s.endswith(suffix):
            s = s[:-len(suffix)].strip()
            break

    return s

def _to_sqlite(csv_buf: io.BytesIO, sqlite_path: str) -> tuple[int, int]:
    """Carrega PID e SONG em um SQLite (em disco), em chunks. Retorna (linhas, playlists_distintas)."""
    conn = sqlite3.connect(sqlite_path)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("CREATE TABLE IF NOT EXISTS plays (pid TEXT, song TEXT);")
    conn.commit()

    total_rows = 0
    for chunk in pd.read_csv(csv_buf, dtype=str, keep_default_na=False,
                             chunksize=CHUNK_ROWS, low_memory=True):
        def pick(cands):
            for n in cands:
                for c in chunk.columns:
                    if c.lower() == n.lower():
                        return c
            return None
        pl_col = pick([PLAYLIST_COL, "playlist_id", "pid", "playlist"])
        sg_col = pick([SONG_COL, "song_name", "track_name", "track", "track_uri", "song"])
        if not pl_col or not sg_col:
            raise ValueError("Não consegui identificar as colunas de playlist e música.")
        sub = chunk[[pl_col, sg_col]].rename(columns={pl_col:"pid", sg_col:"song"})
        rows = [(str(p).strip(), _norm(s)) for p,s in sub.itertuples(index=False) if str(p).strip() and str(s).strip()]
        if not rows:
            continue
        cur.execute("BEGIN")
        cur.executemany("INSERT INTO plays(pid, song) VALUES (?,?)", rows)
        conn.commit()
        total_rows += len(rows)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_pid ON plays(pid);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_song ON plays(song);")
    conn.commit()

    n_playlists = cur.execute("SELECT COUNT(DISTINCT pid) FROM plays;").fetchone()[0]
    conn.close()
    return total_rows, n_playlists

def _dump_tx_and_item_supp(sqlite_path: str, tx_path: str) -> tuple[int, dict]:
    """
    Gera transactions (uma linha por playlist, itens separados por TAB) varrendo ordenado por pid.
    Calcula suporte de 1-item (por #playlists) sem usar group_concat (menos RAM).
    """
    conn = sqlite3.connect(sqlite_path)
    cur = conn.cursor()
    cur_it = cur.execute("SELECT pid, song FROM plays ORDER BY pid;")

    n_baskets = 0
    item_supp = {}
    with open(tx_path, "w", encoding="utf-8") as fout:
        last_pid = None
        seen = set()
        basket = []

        for pid, song in cur_it:
            if last_pid is None:
                last_pid = pid
            if pid != last_pid:
                if len(basket) >= 2:
                    fout.write("\t".join(basket) + "\n")
                    n_baskets += 1
                    for it in seen:
                        item_supp[it] = item_supp.get(it, 0) + 1
                last_pid = pid
                seen.clear()
                basket.clear()

            if song and song not in seen:
                seen.add(song)
                basket.append(song)

        if basket:
            if len(basket) >= 2:
                fout.write("\t".join(basket) + "\n")
                n_baskets += 1
                for it in seen:
                    item_supp[it] = item_supp.get(it, 0) + 1

    conn.close()
    return n_baskets, item_supp

def _filter_transactions(tx_in: str, tx_out: str, item_supp: dict, abs_min_sup: int) -> None:
    """Remove itens com suporte < abs_min_sup, reduzindo MUITO a memória da mineração."""
    keep = {it for it, cnt in item_supp.items() if cnt >= abs_min_sup}
    with open(tx_in, "r", encoding="utf-8") as fin, open(tx_out, "w", encoding="utf-8") as fout:
        for line in fin:
            items = [x for x in line.rstrip("\n").split("\t") if x in keep]
            if len(items) >= 2:
                fout.write("\t".join(items) + "\n")

def _mine_with_fim(tx_path: str, abs_min_sup: int, min_conf: float):

    rules = _fim.fpgrowth(tx_path, target='r', supp=abs_min_sup, conf=int(min_conf*100),
                          report='aC')

    for rule in rules:
        if len(rule) == 3:
            ant, cons, conf = rule
        elif len(rule) == 4:
            ant, cons, supp, conf = rule
        else:
            ant, cons = rule[0], rule[1]
            conf = rule[-1]

        if isinstance(ant, str):
            ant = tuple([x for x in ant.split('\t') if x])
        if isinstance(cons, str):
            cons = tuple([x for x in cons.split('\t') if x])
        yield tuple(ant), tuple(cons), None, float(conf)/100.0

def _mine_pairwise_fallback(tx_path: str, abs_min_sup: int, min_conf: float):

    from collections import Counter
    item_cnt = Counter()
    pair_cnt = Counter()

    with open(tx_path, "r", encoding="utf-8") as f:
        for line in f:
            items = line.rstrip("\n").split("\t")
            if len(items) > 200:
                items = items[:200]
            uniq = list(dict.fromkeys(items))
            for a in uniq:
                item_cnt[a] += 1
            L = len(uniq)
            for i in range(L):
                ai = uniq[i]
                for j in range(i+1, L):
                    bj = uniq[j]
                    if ai <= bj:
                        pair_cnt[(ai,bj)] += 1
                    else:
                        pair_cnt[(bj,ai)] += 1

    for (a,b), c_ab in pair_cnt.items():
        if c_ab >= abs_min_sup:
            conf_ab = c_ab / max(1, item_cnt[a])
            if conf_ab >= min_conf:
                yield (a,), (b,), c_ab, conf_ab
            conf_ba = c_ab / max(1, item_cnt[b])
            if conf_ba >= min_conf:
                yield (b,), (a,), c_ab, conf_ba

def _mine_rules(tx_path: str, abs_min_sup: int, min_conf: float):
    
    return _mine_pairwise_fallback(tx_path, abs_min_sup, min_conf)

    
def _build_rules_map(rules_iter, max_rules_per_ant=30):
    from heapq import nlargest
    tmp = {}
    count = 0
    for ant, cons, supp, conf in rules_iter:
        if count < 5:
            print(f"[ML DEBUG] Rule {count}: ant={ant} (type={type(ant)}), cons={cons} (type={type(cons)}), conf={conf}")

        # Não precisa normalizar aqui, já está normalizado desde o SQLite
        ant_t = tuple(sorted([str(x) for x in ant]))
        cons_t = tuple(sorted([str(x) for x in cons]))

        if count < 5:
            print(f"[ML DEBUG] Converted: ant_t={ant_t}, cons_t={cons_t}")

        tmp.setdefault(ant_t, []).append((cons_t, float(conf)))
        count += 1
    rules_map = {}
    for ant, lst in tmp.items():
        rules_map[ant] = nlargest(max_rules_per_ant, lst, key=lambda x: x[1])
    print(f"[ML] Total de regras brutas processadas: {count}")
    print(f"[ML] Total de antecedentes únicos: {len(rules_map)}")
    if len(rules_map) > 0:
        sample_ant = list(rules_map.keys())[0]
        print(f"[ML] Sample antecedent: {sample_ant}")
    return rules_map

def _save_model(rules_map: dict, n_baskets: int, abs_min_sup: int):
    model = {
        "kind": "fp_rules",
        "rules": rules_map,
        "min_sup_ratio": MIN_SUP,
        "min_conf": MIN_CONF,
        "abs_min_sup": abs_min_sup,
        "dataset_url": DATASET_URL,
        "model_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "song_norm": "casefold",
        "n_baskets": n_baskets,
        "version": "0.2.0-lowmem"
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "kind": model["kind"],
                "antecedents": len(rules_map),
                "min_sup_ratio": model["min_sup_ratio"],
                "min_conf": model["min_conf"],
                "abs_min_sup": model["abs_min_sup"],
                "n_baskets": model["n_baskets"],
                "dataset_url": model["dataset_url"],
                "model_date": model["model_date"],
                "version": model["version"]
            },
            f, ensure_ascii=False, indent=2
        )

# -------------------- Main --------------------
def main():
    print(f"[ML] Lendo dataset: {DATASET_URL}")
    sqlite_path = os.path.join(TMP_DIR, "plays.sqlite")
    buf = _download_or_open(DATASET_URL)
    total_rows, n_playlists = _to_sqlite(buf, sqlite_path)
    print(f"[ML] Linhas inseridas: {total_rows} | Playlists distintas: {n_playlists}")

    tx_all = os.path.join(TMP_DIR, "transactions_all.txt")
    n_baskets, item_supp = _dump_tx_and_item_supp(sqlite_path, tx_all)
    if n_baskets == 0:
        print("[ML] Zero cestas após parsing. Salvando modelo vazio.")
        _save_model({}, 0, 0)
        return
    abs_min_sup = int(math.ceil(MIN_SUP * n_baskets)) if MIN_SUP < 1 else int(MIN_SUP)
    abs_min_sup = max(2, abs_min_sup)
    print(f"[ML] Baskets: {n_baskets} | abs_min_sup: {abs_min_sup}")

    tx_filtered = os.path.join(TMP_DIR, "transactions_filtered.txt")
    _filter_transactions(tx_all, tx_filtered, item_supp, abs_min_sup)

    if _fim is not None:
        print("[ML] Minerando com 'fim' (C) FP-Growth…")
    else:
        print("[ML] 'fim' não encontrado — usando fallback pairwise 1->1 (low-mem).")

    rules_iter = _mine_rules(tx_filtered, abs_min_sup, MIN_CONF)
    rules_map = _build_rules_map(rules_iter, MAX_RULES_PER_ANT)
    print(f"[ML] Antecedentes gerados: {len(rules_map)}")

    _save_model(rules_map, n_baskets, abs_min_sup)
    print(f"[ML] Modelo salvo em: {MODEL_PATH}")
    print(f"[ML] Meta salvo em:   {META_PATH}")

if __name__ == "__main__":
    main()
