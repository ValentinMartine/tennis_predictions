from datetime import datetime, timezone

from src.actualite import (
    archiver,
    lire_rss,
    question_forme,
    signal_joueur,
    url_rss,
)

RSS = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss><channel>
<item><title>Rune returns from Achilles tear - Tennis.com</title><link>https://a/1</link>
<pubDate>Mon, 28 Sep 2026 10:00:00 GMT</pubDate><source url="https://tennis.com">Tennis.com</source></item>
<item><title>Rune vs Jacquet preview - ATP Tour</title><link>https://a/2</link>
<pubDate>Wed, 30 Sep 2026 08:00:00 GMT</pubDate><source url="https://atptour.com">ATP Tour</source></item>
</channel></rss>"""

MAINTENANT = datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)


def test_url_rss_cherche_le_nom_exact_sur_la_periode():
    url = url_rss("Holger Rune", jours=7)
    assert "q=%22Holger+Rune%22+tennis+when%3A7d" in url
    assert url.startswith("https://news.google.com/rss/search?")


def test_lire_rss_garde_titre_date_source_et_lien():
    titres = lire_rss(RSS)
    assert titres[0] == {
        "titre": "Rune returns from Achilles tear - Tennis.com",
        "date": "2026-09-28T10:00:00+00:00",
        "source": "Tennis.com",
        "lien": "https://a/1",
    }
    assert len(titres) == 2


def test_la_question_nomme_le_joueur():
    assert "Holger Rune" in question_forme("Holger Rune")


def test_le_signal_retient_le_titre_le_plus_alarmant_de_la_fenetre():
    evalues = [
        {"titre": "ancien", "date": "2026-09-10T10:00:00+00:00", "p": 0.95},
        {"titre": "retour de blessure", "date": "2026-09-28T10:00:00+00:00", "p": 0.70},
        {"titre": "preview", "date": "2026-09-30T08:00:00+00:00", "p": 0.02},
    ]
    s = signal_joueur(evalues, MAINTENANT, jours=7)
    assert (s["p"], s["titre"], s["n_titres"]) == (0.70, "retour de blessure", 2)


def test_un_titre_posterieur_a_l_instant_de_reference_est_ignore():
    evalues = [{"titre": "futur", "date": "2026-10-01T10:00:00+00:00", "p": 0.99}]
    assert signal_joueur(evalues, MAINTENANT, jours=7) is None


def test_l_archive_ne_duplique_pas_un_titre_deja_vu(tmp_path):
    archive = tmp_path / "titres.jsonl"
    ligne = {"joueur": "Holger Rune", "titre": "t", "lien": "https://a/1", "date": "2026-09-28T10:00:00+00:00", "p": 0.7}
    assert archiver(archive, [ligne]) == 1
    assert archiver(archive, [ligne, {**ligne, "lien": "https://a/2"}]) == 1
    assert len(archive.read_text(encoding="utf-8").splitlines()) == 2


def test_signaux_perimes_ignores(tmp_path):
    import json

    from src.export import charger_signaux

    chemin = tmp_path / "signaux.json"
    alerte = {"p": 0.8, "titre": "t"}
    chemin.write_text(json.dumps({"generated_at": "2026-09-30T08:00:00+00:00", "joueurs": {"Holger Rune": alerte}}))
    assert charger_signaux(chemin, MAINTENANT) == {"Holger Rune": alerte}
    assert charger_signaux(chemin, datetime(2026, 10, 5, tzinfo=timezone.utc)) == {}
    assert charger_signaux(tmp_path / "absent.json", MAINTENANT) == {}


def test_seules_les_alertes_au_dessus_du_seuil_sont_exportees():
    from src.export import alerte_actualite

    signaux = {"A": {"p": 0.72, "titre": "blessé"}, "B": {"p": 0.2, "titre": "preview"}}
    assert alerte_actualite(signaux, "A") == {"p": 0.72, "titre": "blessé"}
    assert alerte_actualite(signaux, "B") is None
    assert alerte_actualite(signaux, "C") is None


def test_l_etiquetage_conserve_les_etiquettes_saisies(tmp_path):
    import csv
    import json

    from scripts.signaux_actualite import mettre_a_jour_etiquetage

    archive, fichier = tmp_path / "titres.jsonl", tmp_path / "a_etiqueter.csv"
    lignes = [{"date": "d", "joueur": "A", "titre": "t1", "source": "s", "lien": "l1", "p": 0.9},
              {"date": "d", "joueur": "B", "titre": "t2", "source": "s", "lien": "l2", "p": 0.1}]
    archive.write_text("\n".join(json.dumps(x) for x in lignes[:1]) + "\n", encoding="utf-8")
    mettre_a_jour_etiquetage(archive, fichier)
    texte = fichier.read_text(encoding="utf-8").replace("l1,0.9,", "l1,0.9,oui")
    fichier.write_text(texte, encoding="utf-8")
    archive.write_text("\n".join(json.dumps(x) for x in lignes) + "\n", encoding="utf-8")
    mettre_a_jour_etiquetage(archive, fichier)
    with fichier.open(encoding="utf-8", newline="") as f:
        assert [r["etiquette"] for r in csv.DictReader(f)] == ["oui", ""]


def test_l_etiqueteur_est_conserve(tmp_path):
    import csv
    import json

    from scripts.signaux_actualite import mettre_a_jour_etiquetage

    archive, fichier = tmp_path / "titres.jsonl", tmp_path / "a_etiqueter.csv"
    archive.write_text(json.dumps({"date": "d", "joueur": "A", "titre": "t", "source": "s", "lien": "l", "p": 0.9}) + "\n",
                       encoding="utf-8")
    mettre_a_jour_etiquetage(archive, fichier)
    texte = fichier.read_text(encoding="utf-8").replace("l,0.9,,", "l,0.9,oui,claude")
    fichier.write_text(texte, encoding="utf-8")
    mettre_a_jour_etiquetage(archive, fichier)
    with fichier.open(encoding="utf-8", newline="") as f:
        assert [(r["etiquette"], r["etiqueteur"]) for r in csv.DictReader(f)] == [("oui", "claude")]


def test_calibration_stratifiee():
    from scripts.calibration_actualite import calibrer

    lignes = ([{"p": 0.9, "etiquette": "oui", "strate": "haute"}] * 9 + [{"p": 0.4, "etiquette": "non", "strate": "haute"}]
              + [{"p": 0.01, "etiquette": "non", "strate": "basse"}] * 100)
    bilan = calibrer(lignes, poids={"haute": 1.0, "basse": 10.0}, seuil=0.5)
    assert bilan["precision"] == 1.0
    assert bilan["faux_negatifs_observes"] == 0
    assert bilan["n_etiquetes"] == 110
    assert 0 < bilan["brier_pondere"] < 0.02
