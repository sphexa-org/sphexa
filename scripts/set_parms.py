#!/usr/bin/env python3

# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.o

"""
Command line utility to create an HDF5 simulation parameters file with key-value pairs

Usage examples:
    $ settings.py <parameter_file> [--add] [list of settings as key-value-pairs, e.g --ng0 100]'
"""

__program__ = "set_parms.py"
__author__ = "Sebastian Keller (sebastian.f.keller@gmail.com)"

import os
from argparse import ArgumentParser

import h5py

def group_list(lst, n):
    return [lst[i:i+n] for i in range(0, len(lst), n)]

def parse_value(v):
    if "," in v:
        try:
        # Try parsing as list of floats
            return [int(x) for x in v.split(",")]
        except ValueError:
            pass
        # Try parsing as list of floats
        try:
            return [float(x) for x in v.split(",")]
        except ValueError:
            pass
        # If both fail, keep as a single comma-separated string
        # (not a list, so C++ can read it easily as a string attribute)
        return v
    else:
        # Try parsing as single integer
        try:
            return int(v)
        except ValueError:
            pass
        # Try parsing as single float
        try:
            return float(v)
        except ValueError:
            pass
        # If both fail, keep as string
        return v

if __name__ == "__main__":
    parser = ArgumentParser(description="Create settings file")
    parser.add_argument("settingsFile", help="Simulation settings HDF5 file")
    parser.add_argument("-a", "--add", action="store_true", dest="addSettings", help="add settings if settings file exists")

    args, settings = parser.parse_known_args()

    fmode = "w"
    if (args.addSettings): fmode = "a"

    f = h5py.File(args.settingsFile, mode=fmode)
    settingsDict = dict(zip(settings[:-1:2], settings[1::2]))
    for k, v in settingsDict.items():
        key = k.strip("-")
        v = parse_value(v)
        print("Parsed setting:", key, "=", v)
        if isinstance(v, str):
            # If v is a string, to fixed-length for HDF5 compatibility
            dtype = h5py.string_dtype(encoding="ascii", length=len(v))
            f.attrs.create(key, v, dtype=dtype)
        else:
            f.attrs[key] = v

    print("{0} now contains the following settings:".format(args.settingsFile))
    for k, v in f.attrs.items():
        # Check if it's an array-like object with size attribute
        if hasattr(v, 'size') and v.size > 1:
            v = v.tolist()
            if k == "id_selection_spheres":
                grouped = group_list(v, 4)
                formatted_value = "[" + ", ".join(str(g) for g in grouped) + "]"
            else:
                formatted_value = "[" + ", ".join(map(str, v)) + "]"

            print(f"   {k} {formatted_value}")
        else:
            if isinstance(v, bytes):
                try:
                    v = v.decode("ascii")
                except UnicodeDecodeError:
                    v = v.decode("utf-8", errors="replace")
            print(f"   {k} {v}")
    f.close()
