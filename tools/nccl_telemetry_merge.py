#!/usr/bin/env python3
import sys
out = sys.argv[1]
with open(out, 'w') as dst:
    for path in sys.argv[2:]:
        with open(path, errors='replace') as src:
            for line in src:
                if not line.startswith('HEADER '): dst.write(line)
