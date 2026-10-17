"""Exercise the search demos offline, without replacing their search implementation."""

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("notebook", ["search.ipynb", "spanish/search.ipynb"])
def test_notebook_search_ranks_movies_with_cosine_similarity(notebook):
    cells = json.loads((ROOT / notebook).read_text(encoding="utf-8"))["cells"]
    offline_cells = {
        tag: "".join(cell["source"])
        for cell in cells
        for tag in cell.get("metadata", {}).get("tags", [])
        if tag.startswith("offline-search-")
    }

    # Pythagorean unit vectors give independently known cosine similarities.
    # Insertion order deliberately differs from ranking to catch label/title mixups.
    coordinates = {
        "opposite": (-1, 0),
        "medium": (20 / 29, 21 / 29),
        "orthogonal": (0, 1),
        "closest": (1, 0),
        "far": (-0.6, 0.8),
        "near": (0.8, 0.6),
        "almost closest": (0.96, 0.28),
        "slightly similar": (0.28, 0.96),
        "slightly dissimilar": (-0.28, 0.96),
        "very far": (-0.8, 0.6),
        "almost opposite": (-0.96, 0.28),
    }
    movies = {title: [*xy, *([0] * 1534)] for title, xy in coordinates.items()}
    query = [1, *([0] * 1535)]
    namespace = {"movies": movies, "new_vector": query}

    def execute(tag):
        exec(compile(offline_cells[tag], f"<{notebook}:{tag}>", "exec"), namespace)

    execute("offline-search-functions")
    expected = [
        ("closest", 1),
        ("almost closest", 0.96),
        ("near", 0.8),
        ("medium", 0.69),
        ("slightly similar", 0.28),
        ("orthogonal", 0),
        ("slightly dissimilar", -0.28),
        ("far", -0.6),
        ("very far", -0.8),
        ("almost opposite", -0.96),
        ("opposite", -1),
    ]
    execute("offline-search-exhaustive-results")
    assert [title for title, _ in namespace["similar_movies"]] == [title for title, _ in expected[:10]]
    assert [score for _, score in namespace["similar_movies"]] == pytest.approx(
        [score for _, score in expected[:10]], abs=1e-6
    )

    execute("offline-search-index")
    execute("offline-search-results")
    # These are the actual notebook's converted HNSW distances and movie titles.
    assert [title for title, _ in namespace["similar_movies"]] == [title for title, _ in expected[:10]]
    assert [score for _, score in namespace["similar_movies"]] == pytest.approx(
        [score for _, score in expected[:10]], abs=1e-6
    )
    assert namespace["distances"][0].tolist() == pytest.approx(
        [0, 0.04, 0.2, 0.3103448275862069, 0.72, 1, 1.28, 1.6, 1.8, 1.96], abs=1e-6
    )
