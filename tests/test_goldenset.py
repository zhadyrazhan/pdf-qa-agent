"""Checks the golden set's structure (eval/goldenset.json) — a fast, network-free
sanity check so a JSON typo or missing field is caught in CI, not during eval."""
import json
from pathlib import Path

GOLDENSET_PATH = Path(__file__).resolve().parents[1] / "eval" / "goldenset.json"

REQUIRED_FIELDS = {"id", "pdf", "question", "expected_answerable", "expected_keywords", "expected_pages"}


def _load():
    return json.loads(GOLDENSET_PATH.read_text(encoding="utf-8"))


def test_goldenset_file_is_valid_json_list():
    data = _load()
    assert isinstance(data, list)
    assert len(data) > 0


def test_every_item_has_required_fields():
    for item in _load():
        missing = REQUIRED_FIELDS - item.keys()
        assert not missing, f"{item.get('id', '<no id>')} missing fields: {missing}"


def test_ids_are_unique():
    ids = [item["id"] for item in _load()]
    assert len(ids) == len(set(ids)), "duplicate ids in golden set"


def test_unanswerable_items_have_no_expected_keywords_or_pages():
    for item in _load():
        if not item["expected_answerable"]:
            assert item["expected_keywords"] == []
            assert item["expected_pages"] == []


def test_answerable_items_have_at_least_one_keyword():
    for item in _load():
        if item["expected_answerable"]:
            assert item["expected_keywords"], f"{item['id']} has no expected_keywords"


def test_covers_both_text_layer_and_scanned_documents():
    """The golden set must cover both text-layer and scanned parts of documents,
    or eval won't exercise the agent's required behavior."""
    notes = " ".join(item.get("note", "") for item in _load())
    assert "OCR" in notes
    assert "текстовый слой" in notes
