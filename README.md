# LSH

A small teaching project on finding similar documents with Locality-Sensitive
Hashing. The same pipeline is worked twice: once in Python and once by hand in
a spreadsheet, so every intermediate value can be inspected.

## Files

| File | What it is |
| --- | --- |
| `lsh_pipeline.py` | MinHash + LSH pipeline over the three sample documents. |
| `basketball.txt`, `volleyball.txt`, `photosynthesis.txt` | Sample documents: two related sports texts and one unrelated text. |
| `LSH.xlsx` | The same MinHash + LSH pipeline worked step by step in Excel, one sheet per stage. |
| `SimHash_vectors.xlsx` | A companion exercise on a different algorithm: SimHash over vectors with cosine similarity and random hyperplanes. Not implemented in Python. |

## Usage

The module has no dependencies outside the standard library and produces no
output on its own; it returns the data from every stage.

```python
from lsh_pipeline import run_pipeline

results = run_pipeline()
results["true_similarity"]       # exact Jaccard per pair (the ground truth)
results["estimated_similarity"]  # MinHash estimate per pair
results["candidates"]            # pairs sharing at least one LSH bucket
```

With the default settings (single-word shingles, 128 permutations, 64 bands
of 2 rows) the two sports documents land in shared buckets and become a
candidate pair, while photosynthesis shares no bucket with either and is
never compared.

## Pipeline stages

1. **Shingling** — each document becomes a set of word pieces
   (`SHINGLE_SIZE` consecutive words, stopwords removed first).
2. **True similarity** — exact Jaccard similarity per pair, kept as the
   ground truth the estimates are checked against.
3. **The queue** — one random shuffle of the combined vocabulary, read
   front to back; a document's MinHash is the position of the first piece
   it owns.
4. **MinHash** — the queue repeated over `NUM_PERMUTATIONS` shuffles turns
   each document into a fixed-length fingerprint; the fraction of matching
   positions estimates the Jaccard similarity.
5. **LSH buckets** — fingerprints are split into `BANDS` bands of
   `ROWS_PER_BAND` rows; documents sharing any band's bucket become
   candidate pairs. The collision threshold is roughly
   `(1 / BANDS) ** (1 / ROWS_PER_BAND)`. `BANDS * ROWS_PER_BAND` must equal
   `NUM_PERMUTATIONS`, so change the two together.
