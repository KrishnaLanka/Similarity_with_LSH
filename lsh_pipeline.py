"""Locality-Sensitive Hashing pipeline.

Runs the MinHash and LSH pipeline over a set of text documents and returns
the data produced at every stage:

    1. shingling      each document becomes a set of pieces
    2. jaccard        exact overlap between two sets (the ground truth)
    3. the_queue      one random shuffle read as a line-up
    4. minhash        each document becomes a fixed-length fingerprint
    5. lsh_buckets    fingerprints are banded so lookalikes share a bucket

Import the module and call ``run_pipeline()`` to get a map holding every
stage's output, or call the stage functions individually.
"""

import os
import re
import random
from itertools import combinations
from collections import defaultdict


DOC_FILES = {
    "Basketball": "basketball.txt",
    "Volleyball": "volleyball.txt",
    "Photosynthesis": "photosynthesis.txt",
}

SHINGLE_SIZE = 1
REMOVE_STOPWORDS = True
NUM_PERMUTATIONS = 128
BANDS = 64
ROWS_PER_BAND = 2
RANDOM_SEED = 3

STOPWORDS = set(
    """
    a an the is are was were be been being am of to in on at by for with and or but as it its
    this that these those from into over under up down out off back each other another all any
    some no not can could must will would shall may might do does did has have had who whom which
    what where when while their his her your our my two three four five six seven eight nine ten
    they them he she we you i than then so if because about above below between only own same such
    very just also more most many much both once here there during before after
    """.split()
)


def load_documents(directory):
    """Read every configured document from ``directory`` into a name to text map."""
    documents = {}
    for name, filename in DOC_FILES.items():
        with open(os.path.join(directory, filename), encoding="utf-8") as handle:
            documents[name] = handle.read()
    return documents


def shingling(text, size=SHINGLE_SIZE, remove_stopwords=REMOVE_STOPWORDS):
    """Return the set of overlapping word pieces contained in ``text``.

    ``size`` is the number of consecutive words per piece: 1 yields single
    words, 2 yields word pairs, and so on.

    The text is lowercased and every character other than letters and
    whitespace is dropped, so digits never reach the pipeline. Stopword
    removal happens before pieces are formed, so a multi-word piece may
    bridge removed words and even sentence boundaries.
    """
    words = re.sub(r"[^a-z\s]", " ", text.lower()).split()
    if remove_stopwords:
        words = [word for word in words if word not in STOPWORDS and len(word) > 2]
    if size == 1:
        return set(words)
    return {" ".join(words[i:i + size]) for i in range(len(words) - size + 1)}


def jaccard(left, right):
    """Return the Jaccard similarity of two sets: shared items over total items.

    Two empty sets are taken to have similarity 0.0.
    """
    if not left and not right:
        return 0.0
    return len(left & right) / len(left | right)


def true_similarity(shingle_sets):
    """Return the exact Jaccard similarity for every pair of documents."""
    return {
        (left, right): jaccard(shingle_sets[left], shingle_sets[right])
        for left, right in combinations(shingle_sets, 2)
    }


def build_vocabulary(shingle_sets):
    """Return the sorted list of every distinct piece and a piece to index map."""
    vocabulary = sorted(set().union(*shingle_sets.values()))
    index = {piece: position for position, piece in enumerate(vocabulary)}
    return vocabulary, index


def make_permutations(vocabulary_size, count, seed=RANDOM_SEED):
    """Return ``count`` random shuffles of the vocabulary.

    Each shuffle is a list of vocabulary indices in line-up order, so element
    ``order[position]`` is the piece standing at that position.
    """
    generator = random.Random(seed)
    permutations = []
    for _ in range(count):
        order = list(range(vocabulary_size))
        generator.shuffle(order)
        permutations.append(order)
    return permutations


def ranks_of(order):
    """Return the inverse of a shuffle: ``rank[index]`` is the piece's position."""
    rank = [0] * len(order)
    for position, index in enumerate(order):
        rank[index] = position
    return rank


def the_queue(shingle_sets, order, vocabulary, index):
    """Return each document's MinHash under a single shuffle.

    A document's MinHash is the position of the first piece it owns as the
    shuffle is read from front to back. The returned map holds, per document,
    the stopping position and the piece found there.
    """
    rank = ranks_of(order)
    result = {}
    for name, pieces in shingle_sets.items():
        stop = min(rank[index[piece]] for piece in pieces)
        result[name] = (stop, vocabulary[order[stop]])
    return result


def minhash(shingle_sets, index, permutations):
    """Return a MinHash fingerprint for every document.

    Position ``j`` of a fingerprint is the document's MinHash under shuffle
    ``j``: the smallest rank among the pieces the document contains.
    """
    ranks = [ranks_of(order) for order in permutations]
    signatures = {}
    for name, pieces in shingle_sets.items():
        indices = [index[piece] for piece in pieces]
        signatures[name] = [min(rank[i] for i in indices) for rank in ranks]
    return signatures


def estimate_similarity(signatures):
    """Return each pair's estimated similarity: the fraction of matching positions."""
    length = len(next(iter(signatures.values())))
    estimates = {}
    for left, right in combinations(signatures, 2):
        agree = sum(1 for a, b in zip(signatures[left], signatures[right]) if a == b)
        estimates[(left, right)] = agree / length
    return estimates


def lsh_buckets(signatures, bands=BANDS, rows=ROWS_PER_BAND):
    """Return candidate pairs and the bucket map produced by banding.

    Each fingerprint is split into ``bands`` bands of ``rows`` numbers; every
    band is hashed to a bucket. Documents sharing any bucket are candidate
    pairs, keyed by the number of bands in which they collide. ``bands`` and
    ``rows`` must be changed together so their product stays equal to the
    fingerprint length.
    """
    length = len(next(iter(signatures.values())))
    if bands * rows != length:
        raise ValueError(
            f"BANDS * ROWS_PER_BAND ({bands} * {rows} = {bands * rows}) "
            f"must equal the fingerprint length ({length})."
        )
    buckets = defaultdict(list)
    for name, signature in signatures.items():
        for band in range(bands):
            key = (band, tuple(signature[band * rows:(band + 1) * rows]))
            buckets[key].append(name)
    shared = defaultdict(int)
    for members in buckets.values():
        if len(members) > 1:
            for left, right in combinations(sorted(members), 2):
                shared[(left, right)] += 1
    return dict(shared), buckets


def run_pipeline(directory=None):
    """Run the full pipeline and return the data produced at every step."""
    if directory is None:
        directory = os.path.dirname(os.path.abspath(__file__))
    documents = load_documents(directory)
    shingle_sets = {name: shingling(text) for name, text in documents.items()}
    truth = true_similarity(shingle_sets)
    vocabulary, index = build_vocabulary(shingle_sets)
    permutations = make_permutations(len(vocabulary), NUM_PERMUTATIONS)
    queue = the_queue(shingle_sets, permutations[0], vocabulary, index)
    signatures = minhash(shingle_sets, index, permutations)
    estimates = estimate_similarity(signatures)
    candidates, buckets = lsh_buckets(signatures)
    return {
        "documents": documents,
        "shingle_sets": shingle_sets,
        "vocabulary": vocabulary,
        "true_similarity": truth,
        "queue": queue,
        "signatures": signatures,
        "estimated_similarity": estimates,
        "candidates": candidates,
        "buckets": buckets,
    }


if __name__ == "__main__":
    run_pipeline()
