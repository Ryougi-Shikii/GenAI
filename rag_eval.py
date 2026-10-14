
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.WARNING,               # keep the eval output readable
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)

# --------------------------------------------------------------- corpus
CORPUS: dict[str, str] = {
    "python_set_operations.md": """# Python Set Operations

    Python's built-in `set` type is an unordered collection of unique, hashable
    elements. It supports four core mathematical set operations, each available
    both as an operator and as a method.

    ## Union
    Returns every element from either set, duplicates removed.
    - Operator: `a | b`
    - Method: `a.union(b)`

    ## Intersection
    Returns only elements present in both sets.
    - Operator: `a & b`
    - Method: `a.intersection(b)`

    ## Difference
    Returns elements in the first set but not the second.
    - Operator: `a - b`
    - Method: `a.difference(b)`

    ## Symmetric Difference
    Returns elements in exactly one of the two sets (not in both).
    - Operator: `a ^ b`
    - Method: `a.symmetric_difference(b)`

    The in-place variants `|=`, `&=`, `-=`, `^=` and the update methods
    `update`, `intersection_update`, `difference_update`,
    `symmetric_difference_update` mutate the set rather than returning a new one.

    Because sets are unordered, iteration order is not guaranteed. Set membership
    testing runs in average O(1) time thanks to hashing.
    """,

        "python_frozensets.md": """# Frozensets: Immutable Sets

    A `frozenset` is an immutable, hashable version of `set`. Because it is
    hashable, a `frozenset` can be used as a dictionary key or as an element of
    another set — a regular `set` cannot.

    Construct with `frozenset([1, 2, 3])`. All read-only operations work:
    `union`, `intersection`, `difference`, `symmetric_difference`, `issubset`,
    `issuperset`, `isdisjoint`. The mutating methods (`add`, `remove`, `update`,
    `discard`) do not exist on `frozenset`.

    The hash of a frozenset is derived from its contents, so two frozensets with
    the same elements compare equal and share the same hash. This makes them
    useful as keys in a `dict` of sets, or as elements of an outer set — for
    example `{frozenset({1, 2}), frozenset({3, 4})}`.

    Python caches the hash of a frozenset on first computation, so repeated
    lookups are cheap.
    """,

        "python_set_membership.md": """# Membership Testing and Deduplication

    The `in` operator tests membership in a set in average O(1) time:
    `3 in {1, 2, 3}` returns `True`. This is the primary reason to use a set
    over a list when you need fast lookups.

    The most common idiom for removing duplicate values from any iterable is to
    pass it through a `set`:

        unique = list(set(items))

    Note this destroys order. To preserve first-seen order, use
    `dict.fromkeys(items)` instead.

    `set` also provides the comparison predicates `issubset`, `issuperset`, and
    `isdisjoint`. `a.issubset(b)` is equivalent to `a <= b`;
    `a.isdisjoint(b)` is `True` when the two sets share no elements.

    Because sets rely on hashing, their elements must be hashable — numbers,
    strings, tuples, and frozensets are, but lists and dicts are not.
    """,

        "python_dict_operations.md": """# Dictionary Operations

    A `dict` maps hashable keys to arbitrary values. Unlike a set, a dict stores
    key-value pairs rather than bare keys.

    Common operations: `d.keys()`, `d.values()`, `d.items()`, `d.get(k, default)`,
    `d.pop(k)`, `d.update(other)`, `d.setdefault(k, default)`.

    Dict comprehensions build a dict in one expression:
    `{k: v for k, v in pairs}`.

    You can set a value with `d[key] = value`. Assigning to an existing key
    overwrites it; assigning to a new key appends it. Deletion is
    `del d[key]` or `d.pop(key)`.

    Since Python 3.7, dicts preserve insertion order. A dict is a natural way to
    model an ordered set if you only care about keys — `dict.fromkeys(items)`
    yields an order-preserving deduplicator.
    """,

        "environment_setup.md": """# Setting Up a Python Development Environment

    This guide walks through getting Python ready on a fresh machine.

    1. Install Python from python.org or your package manager. Verify with
    `python --version`.
    2. Create a virtual environment: `python -m venv .venv`.
    3. Activate it: `.venv\\Scripts\\activate` on Windows,
    `source .venv/bin/activate` on Unix.
    4. Upgrade pip: `python -m pip install --upgrade pip`.
    5. Install your dependencies: `pip install -r requirements.txt`.

    Never install packages globally — always use a virtual environment per
    project. To set up additional tooling, install `ruff` for linting and
    `pytest` for tests.

    For reproducible environments, pin versions in `requirements.txt` or use
    `pyproject.toml` with `uv` or `poetry`. This sets a baseline for everyone
    on the team.
    """,

        "sql_union_intersect.md": """# SQL Set Operations: UNION, INTERSECT, EXCEPT

    SQL provides set operations that combine the result rows of two SELECT
    statements.

    - `UNION` returns distinct rows from both queries.
    - `UNION ALL` returns all rows, keeping duplicates.
    - `INTERSECT` returns rows present in both queries.
    - `EXCEPT` returns rows in the first query but not the second.

    Each query must return the same number of columns with compatible types.
    Column names come from the first query.

    Example:

        SELECT id FROM customers
        UNION
        SELECT id FROM leads;

    This is analogous to Python's `|` operator on sets but operates on tables.
    Unlike Python sets, SQL preserves nothing about ordering unless you add an
    explicit `ORDER BY`.
    """,

        "math_set_theory.md": """# Set Theory: Foundations

    In mathematics, a set is a collection of distinct objects. The fundamental
    binary operations are:

    - Union (A ∪ B): all elements in A or B.
    - Intersection (A ∩ B): all elements in both A and B.
    - Difference (A \\ B): elements in A but not B.
    - Symmetric difference (A △ B): elements in exactly one of A, B.

    These operations satisfy De Morgan's laws:
        ¬(A ∪ B) = ¬A ∩ ¬B
        ¬(A ∩ B) = ¬A ∪ ¬B

    Cardinality |A| counts elements. The power set P(A) is the set of all
    subsets. Cantor's theorem states |P(A)| > |A| for any set.

    Set theory underpins most of modern mathematics, including topology,
    measure theory, and the Zermelo–Fraenkel axioms.
    """,

        "http_status_codes.md": """# HTTP Status Codes

    HTTP responses carry a three-digit status code.

    - 1xx — Informational. 100 Continue, 101 Switching Protocols.
    - 2xx — Success. 200 OK, 201 Created, 204 No Content.
    - 3xx — Redirection. 301 Moved Permanently, 302 Found, 304 Not Modified.
    - 4xx — Client error. 400 Bad Request, 401 Unauthorized, 403 Forbidden,
    404 Not Found, 429 Too Many Requests.
    - 5xx — Server error. 500 Internal Server Error, 502 Bad Gateway,
    503 Service Unavailable.

    404 means the server understood the request but the resource does not exist.
    500 means the server failed while handling a valid request.
""",
}


# --------------------------------------------------------------- queries
# (query, [expected doc filenames]) -- order doesn't matter, presence does
QUERIES: list[tuple[str, list[str]]] = [
    (
        "What are the different set operations in Python?",
        ["python_set_operations.md"],
    ),
    (
        "How do I remove duplicate values from a list?",
        ["python_set_membership.md"],
    ),
    (
        "How do I set up a Python virtual environment?",
        ["environment_setup.md"],
    ),
    (
        "What is the difference between a set and a frozenset?",
        ["python_frozensets.md"],
    ),
    (
        "How do I combine results from two SQL queries?",
        ["sql_union_intersect.md"],
    ),
    (
        "What does symmetric_difference return?",
        ["python_set_operations.md", "Sets in Python.pdf"],
    ),
    (
        "How do dict comprehensions work?",
        ["python_dict_operations.md"],
    ),
    (
        "What is set theory union in mathematics?",
        ["math_set_theory.md"],
    ),
]


# --------------------------------------------------------------- helpers
def _doc_folder() -> Path:
    from rag.config import DOC_FOLDER
    return Path(DOC_FOLDER)


def ensure_corpus() -> list[str]:
    """Write any missing corpus files. Returns names written."""
    folder = _doc_folder()
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for name, content in CORPUS.items():
        path = folder / name
        if not path.exists():
            path.write_text(content, encoding="utf-8")
            written.append(name)
    return written


def remove_corpus() -> list[str]:
    folder = _doc_folder()
    removed = []
    for name in CORPUS:
        path = folder / name
        if path.exists():
            path.unlink()
            removed.append(name)
    return removed


def unique_doc_names(docs) -> list[str]:
    seen, out = set(), []
    for d in docs:
        name = Path(d.metadata.get("source_file", "")).name
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return out


# --------------------------------------------------------------- eval
async def eval_one(query, expected, factory, translator) -> dict:
    """Run translate -> pool -> rerank, return metrics + retrieved list."""
    from rag.rag import _dedupe_docs, _translate_query
    from rag.config import RERANK_TOP_N 

    variants = await _translate_query(translator, query, expansion_n=3)

    base = factory.get_base_ensemble()
    compressor = factory.get_compressor()

    pooled = []
    for name, text in variants.items():
        if name == "hyde":
            continue
        pooled.extend(await base.ainvoke(text))
    pooled.extend(
        await factory.vector_store.asimilarity_search(variants["hyde"], k=10)
    )
    MAX_RERANK_INPUT = 20
    pooled = _dedupe_docs(pooled)
    if len(pooled) > MAX_RERANK_INPUT:                        
        pooled = pooled[:MAX_RERANK_INPUT]
    try:
        reranked = await compressor.acompress_documents(pooled, query)
    except Exception:
        logging.exception("[eval] reranker failed; using pool head")
        reranked = pooled[:RERANK_TOP_N]

    retrieved = unique_doc_names(reranked)
    expected_set = set(expected)

    hit1 = bool(retrieved[:1]) and retrieved[0] in expected_set
    hit3 = bool(set(retrieved[:3]) & expected_set)
    hit8 = bool(set(retrieved[:8]) & expected_set)

    mrr = 0.0
    for rank, name in enumerate(retrieved, 1):
        if name in expected_set:
            mrr = 1.0 / rank
            break

    return {
        "query": query,
        "expected": expected,
        "retrieved": retrieved,
        "pool_size": len(pooled),
        "hit1": hit1,
        "hit3": hit3,
        "hit8": hit8,
        "mrr": mrr,
    }


async def run_eval() -> None:
    from rag.indexing import sync_documents
    from rag.rag import RetrieverFactory
    from rag.config import get_translation_agent

    print("=" * 72)
    print("SYNCING")
    print("=" * 72)
    sync_documents()

    factory = RetrieverFactory.cached()
    translator = get_translation_agent()

    print()
    print("=" * 72)
    print("RUNNING EVAL")
    print("=" * 72)

    results = []
    for q, expected in QUERIES:
        r = await eval_one(q, expected, factory, translator)
        results.append(r)

        print()
        print(f"Q: {q}")
        print(f"   expected: {', '.join(expected)}")
        print(f"   pool: {r['pool_size']} docs  ->  top-5:")
        for i, name in enumerate(r["retrieved"][:5], 1):
            marker = " <-- OK" if name in expected else ""
            print(f"     {i}. {name}{marker}")
        print(
            f"   Hit@1={r['hit1']!s:5}  Hit@3={r['hit3']!s:5}  "
            f"Hit@8={r['hit8']!s:5}  MRR={r['mrr']:.3f}"
        )

    n = len(results)
    agg = {
        "Hit@1": sum(r["hit1"] for r in results) / n,
        "Hit@3": sum(r["hit3"] for r in results) / n,
        "Hit@8": sum(r["hit8"] for r in results) / n,
        "MRR":   sum(r["mrr"] for r in results) / n,
    }

    print()
    print("=" * 72)
    print(f"AGGREGATE over {n} queries")
    print("=" * 72)
    for k, v in agg.items():
        print(f"  {k:6}: {v:.3f}")

    reranker = os.getenv("RERANKER_MODEL", "(default from config)")
    print(f"  reranker: {reranker}")


# --------------------------------------------------------------- main
def main() -> None:
    if "--cleanup" in sys.argv:
        removed = remove_corpus()
        print(f"Removed {len(removed)} test doc(s): {', '.join(removed) or '-'}")
        from rag.indexing import sync_documents
        sync_documents()
        return

    written = ensure_corpus()
    if written:
        print(f"Wrote {len(written)} test doc(s) to document/: {', '.join(written)}")
        print("(run with --cleanup to remove them)")

    asyncio.run(run_eval())


if __name__ == "__main__":
    main()