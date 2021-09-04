import pysvn
import subprocess
import os
import requests


def check_dependencies():
    """
    Installs dependencies for single view mpi model from google research.
    """
    if not os.path.exists("single_view_mpi"):
        pysvn.Client()
        # NOTE: svn is required for now :(
        subprocess.call(
            [
                "svn",
                "export",
                "--force",
                "https://github.com/google-research/google-research/trunk/single_view_mpi",
            ]
        )
    if not os.path.exists("single_view_mpi_full_keras"):
        url = "https://storage.googleapis.com/stereo-magnification-public-files/models/single_view_mpi_full_keras.tar.gz"
        response = requests.get(url, stream=True)
        file = tarfile.open(fileobj=response.raw, mode="r|gz")
        file.extractall(path=".")


def monkeypatch_ctypes():
    """
    MacOS workaround to address issue with PyOpenGl: https://github.com/PixarAnimationStudios/USD/issues/1372
    """
    import os
    import ctypes.util

    uname = os.uname()
    if uname.sysname == "Darwin" and uname.release >= "20.":
        real_find_library = ctypes.util.find_library

        def find_library(name):
            if name in {"OpenGL", "GLUT"}:  # add more names here if necessary
                return f"/System/Library/Frameworks/{name}.framework/{name}"
            return real_find_library(name)

        ctypes.util.find_library = find_library
    return
