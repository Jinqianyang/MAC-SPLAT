import sys
import os.path as path
HERE_PATH = path.normpath(path.dirname(__file__))
DUSt3R_REPO_PATH = path.normpath(path.join(HERE_PATH, '../../dust3r'))
DUSt3R_LIB_PATH = path.join(DUSt3R_REPO_PATH, 'dust3r')
if path.isdir(DUSt3R_LIB_PATH):
    sys.path.insert(0, DUSt3R_REPO_PATH)
else:
    raise ImportError(f"dust3r is not initialized, could not find: {DUSt3R_LIB_PATH}.\n "
                      "Did you forget to run 'git submodule update --init --recursive' ?")
