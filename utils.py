import pysvn
import subprocess
import os
import requests

def check_dependencies():
    if not os.path.exists("single_view_mpi"):
        pysvn.Client()
        # NOTE: svn is required for now :(
        subprocess.call(["svn", "export", "--force", "https://github.com/google-research/google-research/trunk/single_view_mpi"])
    if not os.path.exists("single_view_mpi_full_keras"):
        url = "https://storage.googleapis.com/stereo-magnification-public-files/models/single_view_mpi_full_keras.tar.gz"
        response = requests.get(url, stream=True)
        file = tarfile.open(fileobj=response.raw, mode="r|gz")
        file.extractall(path=".")