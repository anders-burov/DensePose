import fnmatch
import os

def find_files(_dir, ext='*.jpg'):
    # recursively finds files with stated ext. returns full file names.
    matches = []
    for root, dirnames, filenames in os.walk(_dir):
        for filename in fnmatch.filter(filenames, ext):
            matches.append(os.path.join(root, filename))
    return matches