"""
RandoGPX API Server
-------------------
POST /api/trace/communes
  Body: multipart/form-data avec le fichier GPX
  Retourne: JSON { "communes": [{code_insee, nom, code_postal, departement, region, lat, lon, segment_index}, ...] }
GET /api/commune/{code_insee}
  Retourne: JSON avec toutes les infos de la commune (nom, population, score attractivite, etc.)
GET /api/health
"""

import os, json, sqlite3, xml.etree.ElementTree as ET
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import sqlite3

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).parent
DB_PATH = os.environ.get("DATATOURISME_DB", str(BASE_DIR / "datatourisme.db"))

# ─── GPX Parser ───────────────────────────────────────────────────────────────

def parse_gpx(xml_content: str) -> list[tuple[float, float, int]]:
    """
    Parse un GPX XML et retourne la liste des points (lat, lon, index).
    Ne garde que les points <trkseg><trkpt>.
    """
    root = ET.fromstring(xml_content)
    # Try multiple common GPX namespaces
    for ns in ['{http://www.topografx.com/GPX/1/1}', '{http://www.topografx.com/2008/gpx}',
               '{http://topografix.com/GPX/1/1}', '']:
        pts = root.findall(f'.//{ns}trkpt') or root.findall(f'.//{ns}wpt')
        if pts:
            break
    points = []
    for i, trkpt in enumerate(pts):
        lat = float(trkpt.get('lat'))
        lon = float(trkpt.get('lon'))
        points.append((lat, lon, i))
    return points


# ─── Commune lookup via geo.api.gouv.fr ───────────────────────────────────────

def find_commune_by_coords(lat: float, lon: float) -> dict | None:
    """
    Geocodage inversé via l'API nationale française.
    Retourne {code_insee, nom, code_postal, departement, region} ou None.
    """
    import requests
    try:
        r = requests.get(
            "https://geo.api.gouv.fr/communes",
            params={"lat": lat, "lon": lon, "fields": "code,nom,codesPostaux,departement,region"},
            timeout=5
        )
        if r.status_code == 200 and r.json():
            data = r.json()[0]
            return {
                "code_insee": data["code"],
                "nom": data["nom"],
                "codes_postaux": data.get("codesPostaux", []),
                "departement": data.get("departement", {}).get("nom", ""),
                "region": data.get("region", {}).get("nom", ""),
            }
    except Exception:
        pass
    return None


# ─── Local SQLite lookup (fallback via KD-tree) ────────────────────────────────

def _build_kdtree():
    """Construit un KD-tree simple (liste triée) pour nearest-neighbor local."""
    import math
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT code_insee, nom_standard, code_postal, latitude_centre, longitude_centre FROM communes WHERE latitude_centre IS NOT NULL"
    ).fetchall()
    conn.close()
    return [(r[0], r[1], r[2], r[3], r[4]) for r in rows]

_KDTREE = None

def get_kdtree():
    global _KDTREE
    if _KDTREE is None:
        _KDTREE = _build_kdtree()
    return _KDTREE

def nearest_commune(lat: float, lon: float, max_km: float = 15.0) -> dict | None:
    """Trouve la commune la plus proche via KD-tree sur les centres. max_km = tolérance."""
    import math
    tree = get_kdtree()
    best = None
    best_d = max_km
    for code_insee, nom, cp, lat_c, lon_c in tree:
        d = math.sqrt((lat - lat_c)**2 + (lon - lon_c)**2) * 111  # ~km
        if d < best_d:
            best_d = d
            best = (code_insee, nom, cp, lat_c, lon_c)
    if best:
        return {"code_insee": best[0], "nom": best[1], "code_postal": best[2]}
    return None


# ─── DB helpers ────────────────────────────────────────────────────────────────

def get_commune_details(code_insee: str) -> dict | None:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Attractivite
    row = cur.execute(
        "SELECT * FROM attractivite WHERE code_insee = ?", (code_insee,)
    ).fetchone()

    # Commune data
    com = cur.execute(
        "SELECT * FROM communes WHERE code_insee = ?", (code_insee,)
    ).fetchone()

    # Top POIs
    pois = cur.execute(
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


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "db": DB_PATH})


@app.route("/api/trace/communes", methods=["POST", "OPTIONS"])
@cross_origin(origins="*")
def trace_communes():
    """Reçoit un fichier GPX et retourne les communes traversées."""
    import time, sys
    t0 = time.time()
    t_debug = lambda msg: print(f"[{time.time()-t0:.2f}s] {msg}", file=sys.stderr)

    t_debug(f"start files={list(request.files.keys())} data_len={len(request.data)}")
    if "gpx" not in request.files and not request.data:
        err = {"error": "Aucun fichier GPX fourni", "files": list(request.files.keys()), "data_len": len(request.data)}
        t_debug(f"400: {err}")
        return jsonify(err), 400

    if request.files and "gpx" in request.files:
        gpx_content = request.files["gpx"].read().decode("utf-8")
    else:
        gpx_content = request.data.decode("utf-8")
    t_debug(f"file received, size={len(gpx_content)} bytes")

    try:
        points = parse_gpx(gpx_content)
    except ET.ParseError as e:
        t_debug(f"400: GPX XML invalide: {e}")
        return jsonify({"error": "GPX XML invalide"}), 400

    if not points:
        t_debug("400: Aucun point trouvé dans le GPX")
        return jsonify({"error": "Aucun point trouvé dans le GPX"}), 400
    t_debug(f"parsed {len(points)} points")

    import math
    SAMPLE_M = 100
    sampled = [points[0]]
    for pt in points[1:]:
        prev = sampled[-1]
        d = math.sqrt((pt[0]-prev[0])**2 + (pt[1]-prev[1])**2) * 111_000
        if d >= SAMPLE_M:
            sampled.append(pt)
    if sampled[-1] != points[-1]:
        sampled.append(points[-1])
    t_debug(f"sampled {len(sampled)} points")

    seen_insee = {}
    for lat, lon, idx in sampled:
        commune = nearest_commune(lat, lon, max_km=20.0)
        if commune:
            seen_insee[commune["code_insee"]] = commune
    t_debug(f"kd-tree done: {len(seen_insee)} communes")

    communes_list = list(seen_insee.values())

    # ── Parse custom weights from query params (default: patrimoine×8)
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
            nb_pat, nb_nat, nb_loi, nb_eve, nb_res, nb_heb, pop, poi, lat, lon = row
            dyn_score = (
                (nb_pat or 0)  * WEIGHTS["patrimoine"]  +
                (nb_nat or 0)  * WEIGHTS["nature"]      +
                (nb_loi or 0)  * WEIGHTS["loisirs"]     +
                (nb_eve or 0)  * WEIGHTS["evenement"]    +
                (nb_res or 0)  * WEIGHTS["restauration"] +
                (nb_heb or 0)  * WEIGHTS["hebergement"])
            c |= {
                "population": pop,
                "poi_count": poi,
                "latitude": lat,
                "longitude": lon,
                "attractivite_score": round(dyn_score, 1),
                "score_breakdown": {
                    "patrimoine": f"{(nb_pat or 0)}×{WEIGHTS['patrimoine']}={nb_pat*WEIGHTS['patrimoine'] if nb_pat else 0}",
                    "nature":     f"{(nb_nat or 0)}×{WEIGHTS['nature']}={nb_nat*WEIGHTS['nature'] if nb_nat else 0}",
                    "loisirs":    f"{(nb_loi or 0)}×{WEIGHTS['loisirs']}={nb_loi*WEIGHTS['loisirs'] if nb_loi else 0}",
                    "evenement":  f"{(nb_eve or 0)}×{WEIGHTS['evenement']}={nb_eve*WEIGHTS['evenement'] if nb_eve else 0}",
                    "restauration": f"{(nb_res or 0)}×{WEIGHTS['restauration']}={nb_res*WEIGHTS['restauration'] if nb_res else 0}",
                    "hebergement": f"{(nb_heb or 0)}×{WEIGHTS['hebergement']}={nb_heb*WEIGHTS['hebergement'] if nb_heb else 0}",
                }
            }
        enriched.append(c)
    conn.close()

    # Tri par ordre d'apparition sur la trace (déjà donné par seen_insee)
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
    """Top communes par score d'attractivité."""
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
    row = conn.execute(
        "SELECT * FROM attractivite WHERE code_postal = ?", (code_postal,)
    ).fetchone()
    conn.close()
    if not row:
        return jsonify({"error": "Code postal non trouvé"}), 404
    return jsonify(dict(row))


@app.route("/api/pois", methods=["GET"])
def pois_search():
    """Recherche de POIs autour d'un point (lat, lon, rayon en km) ou par commune."""
    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)
    radius = request.args.get("radius", default=5.0, type=float)  # km
    code_insee = request.args.get("code_insee")
    limit = min(int(request.args.get("limit", 50)), 200)
    poi_type = request.args.get("type")  # patrimoine, nature, restauration, etc.

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    if code_insee:
        rows = cur.execute(
            f"SELECT * FROM pois WHERE code_insee = ? {'AND sous_type LIKE ?' if poi_type else ''} LIMIT ?",
            (code_insee, f"%{poi_type}%" if poi_type else None, limit) if poi_type else (code_insee, limit)
        ).fetchall()
    elif lat is not None and lon is not None:
        # bounding box approximate (1° lat ≈ 111km, 1° lon ≈ 111*cos(lat)km)
        import math
        dlat = radius / 111.0
        dlon = radius / (111.0 * math.cos(math.radians(lat)))
        rows = cur.execute(
            "SELECT * FROM pois WHERE latitude BETWEEN ? AND ? AND longitude BETWEEN ? AND ? LIMIT ?",
            (lat - dlat, lat + dlat, lon - dlon, lon + dlon, limit * 3)
        ).fetchall()
        # Filter by actual distance
        import math as m
        def dist(r):
            return m.sqrt((r['latitude']-lat)**2 + (r['longitude']-lon)**2) * 111
        rows = [r for r in rows if dist(r) <= radius][:limit]
    else:
        return jsonify({"error": "Fournir lat/lon ou code_insee"}), 400

    conn.close()
    return jsonify([dict(r) for r in rows])


# ─── Start ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting RandoGPX API on port {port}")
    print(f"Database: {DB_PATH}")
    app.run(host="0.0.0.0", port=port, debug=False)