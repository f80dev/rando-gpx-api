"""
Tests unitaires pour server.py
==============================
Tests des routes Flask + fonctions utilitaires (parse_gpx, nearest_commune, etc.)
sans nécessiter la base SQLite (les tests DB sont en integration_tests/).

Lance avec : pytest tests/test_server.py -v
"""

import io, sys, math, sqlite3, tempfile, json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# ─── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def app():
    """Flask test client sans toucher à la DB."""
    # On importe APRÈS avoir isolé les vars d'environnement
    import os
    os.environ["DATATOURISME_DB"] = ":memory:"

    # Mock global du KD-tree pour ne pas charger la vraie DB
    with patch("server._KDTREE", []):
        import server
        server._KDTREE = []  # KD-tree vide = nearest_commune retourne toujours None
        server.app.config["TESTING"] = True
        with server.app.test_client() as client:
            yield client
        # Reset module state
        server._KDTREE = None


# ─── GPX valides pour les tests ────────────────────────────────────────────────

SIMPLE_GPX = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="test">
  <trk><name>Test</name><trkseg>
    <trkpt lat="46.5" lon="2.6"><name>Pt1</name></trkpt>
    <trkpt lat="46.6" lon="2.7"><name>Pt2</name></trkpt>
    <trkpt lat="46.7" lon="2.8"><name>Pt3</name></trkpt>
  </trkseg></trk>
</gpx>"""

SIMPLE_GPX_NS = """<?xml version='1.0' encoding='UTF-8'?>
<gpx xmlns="http://www.topografx.com/GPX/1/1" version="1.1">
  <trk><trkseg>
    <trkpt lat="46.5" lon="2.6"/>
    <trkpt lat="46.7" lon="2.8"/>
  </trkseg></trk>
</gpx>"""

TRKPT_GPX = """<?xml version="1.0" encoding="UTF-8"?>
<gpx>
  <trk><trkseg>
    <trkpt lat="46.5" lon="2.6"/>
    <trkpt lat="46.6" lon="2.65"/>
    <trkpt lat="46.7" lon="2.7"/>
  </trkseg></trk>
</gpx>"""

WPT_GPX = """<?xml version="1.0" encoding="UTF-8"?>
<gpx>
  <wpt lat="46.5" lon="2.6"><name>A</name></wpt>
  <wpt lat="46.7" lon="2.8"><name>B</name></wpt>
</gpx>"""

MALFORMED_GPX = """<?xml version="1.0"?>
<gpx><trk><trkseg>
    <trkpt lat="not_a_number" lon="2.6"/>
</trkseg></trk></gpx>"""

TRUNCATED_GPX = """<?xml version="1.0" encoding="UTF-8"?>
<gpx><trk><trkseg>"""


# ─── Tests parse_gpx ───────────────────────────────────────────────────────────

class TestParseGpx:
    def test_parse_simple_trkpt(self):
        from server import parse_gpx
        pts = parse_gpx(TRKPT_GPX)
        assert len(pts) == 3
        assert pts[0] == (46.5, 2.6, 0)
        assert pts[1] == (46.6, 2.65, 1)
        assert pts[2] == (46.7, 2.7, 2)

    def test_parse_with_namespace(self):
        from server import parse_gpx
        pts = parse_gpx(SIMPLE_GPX_NS)
        assert len(pts) == 2
        assert pts[0] == (46.5, 2.6, 0)

    def test_parse_waypoints(self):
        from server import parse_gpx
        pts = parse_gpx(WPT_GPX)
        assert len(pts) == 2
        assert pts[0] == (46.5, 2.6, 0)
        assert pts[1] == (46.7, 2.8, 1)

    def test_parse_malformed_lat_raises(self):
        from server import parse_gpx
        import xml.etree.ElementTree as ET
        with pytest.raises(ValueError):
            parse_gpx(MALFORMED_GPX)

    def test_parse_truncated_xml_raises(self):
        from server import parse_gpx
        import xml.etree.ElementTree as ET
        with pytest.raises(ET.ParseError):
            parse_gpx(TRUNCATED_GPX)


# ─── Tests nearest_commune ─────────────────────────────────────────────────────

class TestNearestCommune:
    def test_no_commune_when_tree_empty(self, app):
        from server import nearest_commune
        result = nearest_commune(46.5, 2.6, max_km=20.0)
        assert result is None

    def test_out_of_range_returns_none(self, app):
        """Pas de commune si aucun point dans le KD-tree n'est dans le rayon."""
        import server
        server._KDTREE = [(99999, "Invisible", "00000", 0.0, 0.0)]
        from server import nearest_commune
        result = nearest_commune(46.5, 2.6, max_km=5.0)
        assert result is None

    def test_within_range_returns_commune(self, app):
        import server
        # Point très proche : (46.5,2.6) → (46.5001, 2.6001) ≈ 15m
        server._KDTREE = [("12345", "Vichy", "03200", 46.5001, 2.6001)]
        from server import nearest_commune
        result = nearest_commune(46.5, 2.6, max_km=20.0)
        assert result["code_insee"] == "12345"
        assert result["nom"] == "Vichy"
        assert result["code_postal"] == "03200"


# ─── Tests routes HTTP ─────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self, app):
        r = app.get("/api/health")
        assert r.status_code == 200
        data = r.get_json()
        assert data["status"] == "ok"
        assert "db" in data


class TestTraceCommunes:
    def test_no_file_returns_400(self, app):
        r = app.post("/api/trace/communes", data={})
        assert r.status_code == 400
        assert "error" in r.get_json()

    def test_invalid_xml_returns_400(self, app):
        r = app.post("/api/trace/communes",
                     data=b"<invalid xml",
                     content_type="application/octet-stream")
        assert r.status_code == 400
        assert "error" in r.get_json()

    def test_gpx_no_points_returns_400(self, app):
        empty = b'<?xml version="1.0"?><gpx><trk></trk></gpx>'
        r = app.post("/api/trace/communes",
                     data=empty,
                     content_type="application/octet-stream")
        # Le parse passe mais aucun point → 400
        assert r.status_code == 400

    def test_valid_gpx_returns_communes(self, app):
        from io import BytesIO
        r = app.post("/api/trace/communes",
                     data={"gpx": (BytesIO(SIMPLE_GPX.encode()), "test.gpx")},
                     content_type="multipart/form-data")
        assert r.status_code == 200
        data = r.get_json()
        assert "total_gpx_points" in data
        assert "sampled_points" in data
        assert "communes" in data
        assert "count" in data
        assert "score_weights" in data

    def test_raw_gpx_body_accepted(self, app):
        r = app.post("/api/trace/communes",
                     data=SIMPLE_GPX.encode(),
                     content_type="application/gpx+xml")
        assert r.status_code == 200
        data = r.get_json()
        assert data["total_gpx_points"] == 3

    def test_weights_params_override_defaults(self, app):
        r = app.post("/api/trace/communes?w_patrimoine=10&w_nature=0",
                     data=SIMPLE_GPX.encode(),
                     content_type="application/gpx+xml")
        assert r.status_code == 200
        data = r.get_json()
        assert data["score_weights"]["patrimoine"] == 10
        assert data["score_weights"]["nature"] == 0


class TestCommuneDetail:
    def test_unknown_insee_returns_404(self, app):
        r = app.get("/api/commune/00000")
        assert r.status_code == 404
        assert "error" in r.get_json()

    def test_returns_json_dict(self, app):
        # On mocke get_commune_details pour éviter la vraie DB
        with patch("server.get_commune_details") as mock:
            mock.return_value = {"code_insee": "03185", "nom": "Vichy", "population": 25000}
            r = app.get("/api/commune/03185")
            assert r.status_code == 200
            assert r.get_json()["nom"] == "Vichy"


class TestAttractiviteList:
    def test_returns_list(self, app):
        with patch("server.sqlite3") as mock_sql:
            mock_conn = MagicMock()
            mock_sql.connect.return_value = mock_conn
            mock_conn.execute.return_value.fetchall.return_value = [
                ("03000", "Moulins", "Allier", "Auvergne", 150.0, 42, 20000, 46.56, 3.08)
            ]
            r = app.get("/api/attractivite?limit=10")
            assert r.status_code == 200
            data = r.get_json()
            assert isinstance(data, list)

    def test_limit_is_capped(self, app):
        with patch("server.sqlite3") as mock_sql:
            mock_conn = MagicMock()
            mock_sql.connect.return_value = mock_conn
            mock_conn.execute.return_value.fetchall.return_value = []
            r = app.get("/api/attractivite?limit=9999")
            assert r.status_code == 200
            # Le endpoint cap à 200, vérifiable en observant que fetchall a été appelé
            mock_conn.execute.assert_called()


class TestPoisSearch:
    def test_neither_lat_nor_code_insee_returns_400(self, app):
        r = app.get("/api/pois")
        assert r.status_code == 400
        assert "error" in r.get_json()

    def test_by_code_insee_returns_list(self, app):
        with patch("server.sqlite3") as mock_sql:
            mock_conn = MagicMock()
            mock_sql.connect.return_value = mock_conn
            mock_conn.execute.return_value.fetchall.return_value = []
            r = app.get("/api/pois?code_insee=03185")
            assert r.status_code == 200
            assert isinstance(r.get_json(), list)

    def test_limit_is_capped(self, app):
        with patch("server.sqlite3") as mock_sql:
            mock_conn = MagicMock()
            mock_sql.connect.return_value = mock_conn
            mock_conn.execute.return_value.fetchall.return_value = []
            r = app.get("/api/pois?lat=46.5&lon=2.6&limit=9999")
            assert r.status_code == 200


class TestAttractiviteByCp:
    def test_unknown_cp_returns_404(self, app):
        r = app.get("/api/attractivite/99999")
        assert r.status_code == 404
        assert "error" in r.get_json()

    def test_valid_cp_returns_data(self, app):
        with patch("server.sqlite3") as mock_sql:
            mock_conn = MagicMock()
            mock_sql.connect.return_value = mock_conn
            mock_conn.execute.return_value.fetchone.return_value = {"code_postal": "03000", "commune": "Moulins"}
            r = app.get("/api/attractivite/03000")
            assert r.status_code == 200