"""
RandoGPX API Server — Optimisé v2
--------------------------------
POST /api/trace/communes
  Body: multipart/form-data avec le fichier GPX
GET /api/health

Optimisations:
  - Option 4: coordonnées cartésiennes (x,y,z) pré-calculées à l'init
  - Option 2: R-tree SQLite pour requêtes par boîte englobante
  - Option 3: batching des requêtes R-tree par chunks
"""

import os, json, sqlite3, xml.etree.ElementTree as ET, math, sys, time
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).parent
DB_PATH = os.environ.get("DATATOURISME_DB", str(BASE_DIR / "datatourisme.db"))

# ─── Constantes ─────────────────────────────────────────────────────────────

EARTH_RADIUS_KM = 6371.0
DEG_TO_RAD = math.pi / 180.0
KM_TO_DEG_LAT = 1.0 / 111.0
RADIUS_DEG_EXTRA = 0.01
MAX_KM = 20.0

# ─── Init KD-tree (Option 4) ────────────────────────────────────────────────

_KDTREE = None   # [(rowid, code_insee, nom, cp, lat, lon, x, y, z), ...]
_KDTREE_IDX = {}  # rowid → index dans _KDTREE


def _init_kdtree():
    global _KDTREE, _KDTREE_IDX
    if _KDTREE is not None:
        return

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT rowid, code_insee, nom_standard, code_postal, latitude_centre, longitude_centre
        FROM communes
        WHERE latitude_centre IS NOT NULL AND longitude_centre IS NOT NULL
    """).fetchall()
    conn.close()

    kdtree = []
    for rowid, code_insee, nom, cp, lat_c, lon_c in rows:
        lat_r = lat_c * DEG_TO_RAD
        lon_r = lon_c * DEG_TO_RAD
        cos_lat = math.cos(lat_r)
        kdtree.append((
            rowid, code_insee, nom, cp, lat_c, lon_c,
            cos_lat * math.cos(lon_r),   # x
            cos_lat * math.sin(lon_r),   # y
            math.sin(lat_r)              # z
        ))
        # _KDTREE_IDX keyed by code_insee (string) — R-tree id is the equivalent integer
        _KDTREE_IDX[code_insee] = len(kdtree) - 1

    _KDTREE = kdtree
    print(f"[init] KD-tree: {len(kdtree)} communes", flush=True)


def get_kdtree():
    _init_kdtree()
    return _KDTREE


# ─── R-tree init (Option 2) ─────────────────────────────────────────────────

_RTREE_CONN = None


def _get_rtree_conn():
    global _RTREE_CONN
    if _RTREE_CONN is None:
        _RTREE_CONN = sqlite3.connect(DB_PATH, check_same_thread=False)
    _init_kdtree()
    return _RTREE_CONN


def _ensure_rtree():
    """Rebuild R-tree: DELETE + re-INSERT (preserves table structure)."""
    conn = sqlite3.connect(DB_PATH)

    # If table exists but has wrong id type (rowid=int vs code_insee=str),
    # we must DROP and recreate. Dropping rtree ALSO drops its 3 aux tables.
    try:
        sample = conn.execute("SELECT id FROM communes_rtree LIMIT 1").fetchone()
        if sample is not None and isinstance(sample[0], str):
            # Already has code_insee string ids — just repopulate
            conn.execute("DELETE FROM communes_rtree")
            conn.commit()
        else:
            # Wrong type or empty — must recreate table structure
            for tbl in ['communes_rtree', 'communes_rtree_node',
                       'communes_rtree_parent', 'communes_rtree_rowid']:
                conn.execute(f"DROP TABLE IF EXISTS {tbl}")
            conn.commit()
            conn.execute("""
                CREATE VIRTUAL TABLE communes_rtree USING rtree(
                    id, lat_min, lat_max, lon_min, lon_max
                )
            """)
    except Exception:
        # Table doesn't exist at all — create it
        for tbl in ['communes_rtree', 'communes_rtree_node',
                   'communes_rtree_parent', 'communes_rtree_rowid']:
            conn.execute(f"DROP TABLE IF EXISTS {tbl}")
        conn.commit()
        conn.execute("""
            CREATE VIRTUAL TABLE communes_rtree USING rtree(
                id, lat_min, lat_max, lon_min, lon_max
            )
        """)

    # CAST to TEXT ensures code_insee strings are stored as-is in R-tree id even when
    # SQLite internally handles them as integer (in Metro, all codes are 5-digit).
    # Non-numeric overseas codes (e.g. '2A001') are stored as their integer prefix —
    # but since there are no duplicate code_insee values, no collision occurs.
    conn.execute("""
        INSERT OR IGNORE INTO communes_rtree(id, lat_min, lat_max, lon_min, lon_max)
        SELECT CAST(code_insee AS TEXT),
               latitude_centre - 0.005, latitude_centre + 0.005,
               longitude_centre - 0.005, longitude_centre + 0.005
        FROM communes
        WHERE latitude_centre IS NOT NULL AND longitude_centre IS NOT NULL
    """)
    conn.commit()
    cnt = conn.execute("SELECT COUNT(*) FROM communes_rtree").fetchone()[0]
    conn.close()
    print(f"[init] R-tree rebuilt: {cnt} rows (code_insee as string id)", flush=True)


# ─── Core: nearest commune via R-tree bbox + cartésien ──────────────────────

def _nearest_from_candidates(lat: float, lon: float, candidate_ids: list, max_km: float = MAX_KM) -> dict | None:
    """Trouve la commune la plus proche parmi une liste de code_insee (int)."""
    tree = get_kdtree()
    lat_r = lat * DEG_TO_RAD
    lon_r = lon * DEG_TO_RAD
    qx = math.cos(lat_r) * math.cos(lon_r)
    qy = math.cos(lat_r) * math.sin(lon_r)
    qz = math.sin(lat_r)

    best_d = max_km
    best = None
    for code_insee_int in candidate_ids:
        code_insee_str = str(code_insee_int)
        if code_insee_str not in _KDTREE_IDX:
            continue
        idx = _KDTREE_IDX[code_insee_str]
        _, _, nom, cp, lat_c, lon_c, _, _, _ = tree[idx]
        d = math.sqrt((qx - tree[idx][6])**2 + (qy - tree[idx][7])**2 + (qz - tree[idx][8])**2) * EARTH_RADIUS_KM
        if d < best_d:
            best_d = d
            best = {"code_insee": code_insee_str, "nom": nom, "code_postal": cp, "lat_c": lat_c, "lon_c": lon_c, "dist_km": round(best_d, 2)}
    return best


def _bbox_candidates(lat: float, lon: float, max_km: float = MAX_KM) -> list:
    """Retourne les rowids dans la bbox via R-tree. Une seule requête SQL."""
    conn = _get_rtree_conn()
    dlat = (max_km * KM_TO_DEG_LAT) + RADIUS_DEG_EXTRA
    try:
        dlon = dlat / math.cos(math.radians(lat))
    except Exception:
        dlon = dlat

    cur = conn.execute("""
        SELECT id FROM communes_rtree
        WHERE lat_min <= ? AND lat_max >= ?
          AND lon_min <= ? AND lon_max >= ?
    """, (lat + dlat, lat - dlat, lon + dlon, lon - dlon))
    return [row[0] for row in cur.fetchall()]


# ─── Batch: find_communes_along_trace ───────────────────────────────────────

def find_communes_along_trace(sampled_points: list[tuple[float, float, int]], max_km: float = MAX_KM) -> list[dict]:
    """
    Retourne les communes uniques traversées.
    Stratégie: batch R-tree queries par chunks de points → nearest cartésien.
    """
    t0 = time.time()
    tree = get_kdtree()
    conn = _get_rtree_conn()

    # Phase 1: batch R-tree queries par chunk de 300 points
    all_candidate_ids = set()
    CHUNK = 300

    for chunk_start in range(0, len(sampled_points), CHUNK):
        chunk = sampled_points[chunk_start:chunk_start + CHUNK]
        conditions = []
        params = []
        for lat, lon, _ in chunk:
            dlat = (max_km * KM_TO_DEG_LAT) + RADIUS_DEG_EXTRA
            try:
                dlon = dlat / math.cos(math.radians(lat))
            except Exception:
                dlon = dlat
            conditions.append("(r.lat_min <= ? AND r.lat_max >= ? AND r.lon_min <= ? AND r.lon_max >= ?)")
            params.extend([lat + dlat, lat - dlat, lon + dlon, lon - dlon])

        sql = f"""SELECT DISTINCT r.id FROM communes_rtree r WHERE {" OR ".join(conditions)}"""
        try:
            all_candidate_ids.update(row[0] for row in conn.execute(sql, params).fetchall())
        except Exception:
            # Sous-divise en 100 si trop de termes
            for sub_start in range(0, len(chunk), 100):
                sub = chunk[sub_start:sub_start + 100]
                sub_cond = []
                sub_params = []
                for lat, lon, _ in sub:
                    dlat2 = (max_km * KM_TO_DEG_LAT) + RADIUS_DEG_EXTRA
                    try:
                        dlon2 = dlat2 / math.cos(math.radians(lat))
                    except Exception:
                        dlon2 = dlat2
                    sub_cond.append("(r.lat_min <= ? AND r.lat_max >= ? AND r.lon_min <= ? AND r.lon_max >= ?)")
                    sub_params.extend([lat + dlat2, lat - dlat2, lon + dlon2, lon - dlon2])
                sql2 = f"""SELECT DISTINCT r.id FROM communes_rtree r WHERE {" OR ".join(sub_cond)}"""
                all_candidate_ids.update(row[0] for row in conn.execute(sql2, sub_params).fetchall())

    t1 = time.time()

    # Pré-build code_insee string → kdtree index
    code_insee_to_idx = {str(_id): _KDTREE_IDX[str(_id)] for _id in all_candidate_ids if str(_id) in _KDTREE_IDX}

    # Phase 2: nearest cartésien par point sur tous les candidats
    seen_insee = {}
    t2 = time.time()
    for lat, lon, _ in sampled_points:
        lat_r = lat * DEG_TO_RAD
        lon_r = lon * DEG_TO_RAD
        qx = math.cos(lat_r) * math.cos(lon_r)
        qy = math.cos(lat_r) * math.sin(lon_r)
        qz = math.sin(lat_r)

        best_d = max_km
        best = None
        for code_insee_str, idx in code_insee_to_idx.items():
            _, _, nom, cp, lat_c, lon_c, cx, cy, cz = tree[idx]
            d = math.sqrt((qx - cx)**2 + (qy - cy)**2 + (qz - cz)**2) * EARTH_RADIUS_KM
            if d < best_d:
                best_d = d
                best = {"code_insee": code_insee_str, "nom": nom, "code_postal": cp, "lat_c": lat_c, "lon_c": lon_c}
        if best:
            seen_insee[best["code_insee"]] = best

    t3 = time.time()
    print(f"[find_communes] rtree={t1-t0:.3f}s candidates={len(all_candidate_ids)}/{len(code_insee_to_idx)} nearest={t3-t2:.3f}s total={t3-t0:.3f}s", flush=True)
    return list(seen_insee.values())


# ─── GPX Parser ─────────────────────────────────────────────────────────────

def parse_gpx(xml_content: str) -> list[tuple[float, float, int]]:
    root = ET.fromstring(xml_content)
    for ns in ['{http://www.topografix.com/GPX/1/1}', '{http://www.topografx.com/2008/gpx}', '{http://topografix.com/GPX/1/1}', '']:
        pts = root.findall(f'.//{ns}trkpt') or root.findall(f'.//{ns}wpt')
        if pts:
            break
    return [(float(p.get('lat')), float(p.get('lon')), i) for i, p in enumerate(pts)]


def _sample_gpx_points(points: list[tuple[float, float, int]], sample_m: float = 100.0) -> list[tuple[float, float, int]]:
    if not points:
        return []
    sampled = [points[0]]
    for pt in points[1:]:
        prev = sampled[-1]
        d = math.sqrt((pt[0] - prev[0])**2 + (pt[1] - prev[1])**2) * 111_000
        if d >= sample_m:
            sampled.append(pt)
    if sampled[-1] != points[-1]:
        sampled.append(points[-1])
    return sampled


# ─── DB helpers ─────────────────────────────────────────────────────────────

def get_commune_details(code_insee: str) -> dict | None:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM attractivite WHERE code_insee = ?", (code_insee,)).fetchone()
    com = conn.execute("SELECT * FROM communes WHERE code_insee = ?", (code_insee,)).fetchone()
    pois = conn.execute(
        "SELECT nom, type, sous_type, latitude, longitude FROM pois WHERE code_insee = ? LIMIT 20",
        (code_insee,)
    ).fetchall()
    conn.close()
    if not com:
        return None
    result = dict(com)
    if row:
        result.update(dict(row))
    result["pois"] = [dict(p) for p in pois]
    return result


# ─── Routes ─────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "db": DB_PATH})


@app.route("/api/trace/communes", methods=["POST", "OPTIONS"])
@cross_origin(origins="*")
def trace_communes():
    import time as _time
    t0 = _time.time()
    t_debug = lambda msg: print(f"[{_time.time()-t0:.3f}s] {msg}", file=sys.stderr)

    t_debug(f"start files={list(request.files.keys())} data_len={len(request.data)}")

    if "gpx" not in request.files and not request.data:
        return jsonify({"error": "Aucun fichier GPX fourni"}), 400

    if request.files and "gpx" in request.files:
        gpx_content = request.files["gpx"].read().decode("utf-8")
    else:
        gpx_content = request.data.decode("utf-8")

    try:
        points = parse_gpx(gpx_content)
    except ET.ParseError as e:
        return jsonify({"error": f"GPX XML invalide: {e}"}), 400

    if not points:
        return jsonify({"error": "Aucun point trouvé dans le GPX"}), 400

    sampled = _sample_gpx_points(points, sample_m=100.0)
    t_debug(f"parsed {len(points)} points, sampled {len(sampled)}")

    communes_list = find_communes_along_trace(sampled, max_km=20.0)
    t_debug(f"search done: {len(communes_list)} communes")

    # Parse weights
    try:
        w_pat = int(request.args.get("w_patrimoine", 8))
        w_nat = int(request.args.get("w_nature", 4))
        w_loi = int(request.args.get("w_loisirs", 3))
        w_eve = int(request.args.get("w_evenement", 3))
        w_res = int(request.args.get("w_restauration", 2))
        w_heb = int(request.args.get("w_hebergement", 1))
    except ValueError:
        w_pat, w_nat, w_loi, w_eve, w_res, w_heb = 8, 4, 3, 3, 2, 1
    WEIGHTS = {"patrimoine": w_pat, "nature": w_nat, "loisirs": w_loi,
               "evenement": w_eve, "restauration": w_res, "hebergement": w_heb}

    conn = sqlite3.connect(DB_PATH)
    enriched = []
    for c in communes_list:
        row = conn.execute(
            """SELECT nb_patrimoine, nb_nature, nb_loisirs,
                      nb_evenement, nb_restauration, nb_hebergement,
                      population, poi_count, latitude, longitude
               FROM attractivite WHERE code_insee = ?""",
            (c["code_insee"],)
        ).fetchone()
        if row:
            nb_pat, nb_nat, nb_loi, nb_eve, nb_res, nb_heb, pop, poi, lat_c, lon_c = row
            dyn_score = (
                (nb_pat or 0)  * WEIGHTS["patrimoine"]  +
                (nb_nat or 0)  * WEIGHTS["nature"]      +
                (nb_loi or 0)  * WEIGHTS["loisirs"]     +
                (nb_eve or 0)  * WEIGHTS["evenement"]    +
                (nb_res or 0)  * WEIGHTS["restauration"] +
                (nb_heb or 0)  * WEIGHTS["hebergement"])
            c |= {
                "population": pop, "poi_count": poi,
                "latitude": lat_c, "longitude": lon_c,
                "attractivite_score": round(dyn_score, 1),
                "score_breakdown": {
                    "patrimoine": f"{(nb_pat or 0)}×{WEIGHTS['patrimoine']}={nb_pat*WEIGHTS['patrimoine'] if nb_pat else 0}",
                    "nature": f"{(nb_nat or 0)}×{WEIGHTS['nature']}={nb_nat*WEIGHTS['nature'] if nb_nat else 0}",
                    "loisirs": f"{(nb_loi or 0)}×{WEIGHTS['loisirs']}={nb_loi*WEIGHTS['loisirs'] if nb_loi else 0}",
                    "evenement": f"{(nb_eve or 0)}×{WEIGHTS['evenement']}={nb_eve*WEIGHTS['evenement'] if nb_eve else 0}",
                    "restauration": f"{(nb_res or 0)}×{WEIGHTS['restauration']}={nb_res*WEIGHTS['restauration'] if nb_res else 0}",
                    "hebergement": f"{(nb_heb or 0)}×{WEIGHTS['hebergement']}={nb_heb*WEIGHTS['hebergement'] if nb_heb else 0}",
                }
            }
        enriched.append(c)
    conn.close()

    t_debug(f"total time: {_time.time()-t0:.3f}s")
    return jsonify({
        "total_gpx_points": len(points),
        "sampled_points": len(sampled),
        "communes": enriched,
        "count": len(enriched),
        "score_weights": WEIGHTS,
    })


@app.route("/api/commune/<code_insee>", methods=["GET"])
def commune_detail(code_insee: str):
    data = get_commune_details(code_insee)
    if not data:
        return jsonify({"error": f"Commune {code_insee} non trouvée"}), 404
    return jsonify(data)


@app.route("/api/attractivite", methods=["GET"])
def attractivite_list():
    limit = min(int(request.args.get("limit", 20)), 200)
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT code_postal, commune, departement, region, score, poi_count, population, latitude, longitude "
        "FROM attractivite WHERE latitude IS NOT NULL ORDER BY score DESC LIMIT ?",
        (limit,)
    ).fetchall()
    conn.close()
    return jsonify([{
        "code_postal": r[0], "commune": r[1], "departement": r[2], "region": r[3],
        "score": r[4], "poi_count": r[5], "population": r[6], "lat": r[7], "lon": r[8]
    } for r in rows])


@app.route("/api/attractivite/<code_postal>", methods=["GET"])
def attractivite_by_cp(code_postal: str):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM attractivite WHERE code_postal = ?", (code_postal,)).fetchone()
    conn.close()
    if not row:
        return jsonify({"error": "Code postal non trouvé"}), 404
    return jsonify(dict(row))


@app.route("/api/pois", methods=["GET"])
def pois_search():
    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)
    radius = request.args.get("radius", default=5.0, type=float)
    code_insee = request.args.get("code_insee")
    limit = min(int(request.args.get("limit", 50)), 200)
    poi_type = request.args.get("type")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    if code_insee:
        rows = cur.execute(
            f"SELECT * FROM pois WHERE code_insee = ? {'AND sous_type LIKE ?' if poi_type else ''} LIMIT ?",
            (code_insee, f"%{poi_type}%" if poi_type else None, limit) if poi_type else (code_insee, limit)
        ).fetchall()
    elif lat is not None and lon is not None:
        dlat = radius / 111.0
        try:
            dlon = dlat / math.cos(math.radians(lat))
        except Exception:
            dlon = dlat
        rows = cur.execute(
            "SELECT * FROM pois WHERE latitude BETWEEN ? AND ? AND longitude BETWEEN ? AND ? LIMIT ?",
            (lat - dlat, lat + dlat, lon - dlon, lon + dlon, limit * 3)
        ).fetchall()
        rows = [r for r in rows if math.sqrt((r['latitude']-lat)**2 + (r['longitude']-lon)**2) * 111 <= radius][:limit]
    else:
        conn.close()
        return jsonify({"error": "Fournir lat/lon ou code_insee"}), 400

    conn.close()
    return jsonify([dict(r) for r in rows])


# ─── Start ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time as _time
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting RandoGPX API on port {port}", flush=True)
    print(f"Database: {DB_PATH}", flush=True)
    _ensure_rtree()
    _init_kdtree()
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)