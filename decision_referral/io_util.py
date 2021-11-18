import bz2
import os
import pickle


PACKAGE_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
INPUT_DIR = os.path.join(PACKAGE_ROOT, "data", "inputs")
RESULTS_DIR = os.path.join(PACKAGE_ROOT, "data", "results")


def save_to_pickle(obj: object, filename: str, **kwargs):
    """Save a pickle-able object to a file. If the filename
    ends with .pklz save as a compressed pickle

    Args:
        obj: The object we want to pickle.
        filename: The output filename of the pickled object.
        **kwargs: Arguments passed to `pickle.dump`
    """
    if filename.endswith(".pklz"):
        with bz2.BZ2File(filename, "w") as f:
            pickle.dump(obj, f, **kwargs)
    else:
        with open(filename, "wb") as f:
            pickle.dump(obj, f, **kwargs)


def load_from_pickle(filename: str, **kwargs):
    """Load a pickled object from a file.

    If the filename ends with pklz assumes it's a BZ2 compressed pickle.
    Also, if the ending is pklz, and the file does not
    exist, loads an uncompressed version if that exists.

    Args:
        filename: The filename of the pickled object to load.
        **kwargs: Arguments passed to `pickle.load`

    Returns:
        The unpickled object.
    """
    if filename.endswith(".pklz"):
        with bz2.BZ2File(filename, "r") as f:
            return pickle.load(f, **kwargs)
    else:
        with open(filename, "rb") as f:
            return pickle.load(f, **kwargs)
