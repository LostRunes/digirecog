"""Build the 36-class alphanumeric dataset (0-9, A-Z) from EMNIST 'byclass'.

Usage:  python -I prepare_emnist.py path/to/gzip.zip      (official NIST gzip.zip or the Kaggle archive.zip)
Only the four byclass members are read. Data is parsed as raw bytes
(zip + gzip + numpy.frombuffer) - nothing from the file is executed.

Lower- and upper-case letters are merged (a/A -> A) because several pairs
(c/C, o/O, s/S, ...) are identical in shape; users can draw either case.
Output: data/emnist36.npz  (uint8 images (N,28,28), labels 0..35)
"""
import gzip, os, sys, zipfile
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_PER_CLASS, TEST_PER_CLASS = 5000, 1000


def read_idx(zf, name):
    raw = zf.read(name)
    if raw[0] == 0x1F and raw[1] == 0x8B:   # .gz member (NIST zip) vs raw idx (Kaggle zip)
        raw = gzip.decompress(raw)
    if raw[2] == 0x08:  # unsigned byte
        dims = raw[3]
        shape = tuple(int.from_bytes(raw[4 + 4 * i:8 + 4 * i], "big") for i in range(dims))
        return np.frombuffer(raw, np.uint8, offset=4 + 4 * dims).reshape(shape)
    raise ValueError("unexpected idx type")


def merge(y):
    y = y.astype(np.int64)
    return np.where(y >= 36, y - 26, y)           # a-z (36..61) -> A-Z (10..35)


def balance(x, y, cap, rng):
    keep = []
    for c in range(36):
        idx = np.where(y == c)[0]
        keep.append(rng.choice(idx, min(cap, len(idx)), replace=False))
    keep = rng.permutation(np.concatenate(keep))
    return x[keep], y[keep].astype(np.uint8)


if __name__ == "__main__":
    out = os.path.join(ROOT, "data", "emnist36.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if len(sys.argv) != 2:
        sys.exit("usage: python prepare_emnist.py path/to/gzip.zip")
    zf = zipfile.ZipFile(sys.argv[1])
    names = {os.path.basename(n).removesuffix(".gz"): n for n in zf.namelist()}
    rng = np.random.RandomState(42)
    res = {}
    for split in ("train", "test"):
        print("reading", split, flush=True)
        x = read_idx(zf, names[f"emnist-byclass-{split}-images-idx3-ubyte"])
        y = read_idx(zf, names[f"emnist-byclass-{split}-labels-idx1-ubyte"])
        x = x.transpose(0, 2, 1)                  # EMNIST stores images transposed
        res[split] = balance(x, merge(y), TRAIN_PER_CLASS if split == "train" else TEST_PER_CLASS, rng)
        print(split, res[split][0].shape, flush=True)
    np.savez_compressed(out, x_train=res["train"][0], y_train=res["train"][1],
                        x_test=res["test"][0], y_test=res["test"][1])
    print("saved", out, os.path.getsize(out) / 1e6, "MB")
