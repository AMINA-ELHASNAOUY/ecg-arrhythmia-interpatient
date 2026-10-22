#!/bin/bash
# Resumable, parallel download of the MIT-BIH Arrhythmia Database (skips finished records).
cd "$(dirname "$0")/.."
mkdir -p data/mitdb
BASE=https://physionet.org/files/mitdb/1.0.0
RECS="100 101 102 103 104 105 106 107 108 109 111 112 113 114 115 116 117 118 119 121 122 123 124 200 201 202 203 205 207 208 209 210 212 213 214 215 217 219 220 221 222 223 228 230 231 232 233 234"
for r in $RECS; do
  for ext in hea atr dat; do
    f=data/mitdb/$r.$ext
    if [ "$ext" = dat ] && [ -f $f ] && [ $(stat -f %z $f) -eq 1950000 ]; then continue; fi
    if [ -f $f ] && head -c 6 $f | grep -q "<html"; then rm -f $f; fi   # error page instead of data
    if [ "$ext" = hea ] && [ -s $f ]; then continue; fi
    if [ "$ext" = atr ] && [ -s $f ]; then continue; fi
    echo "$ext $r"
  done
done | xargs -P 8 -n 2 sh -c 'curl -fsS --retry 5 --connect-timeout 20 --max-time 300 -C - -o data/mitdb/$1.$0 '"$BASE"'/$1.$0 && echo "ok $1.$0"'
echo; echo "Complete records:"
n=0; for r in $RECS; do [ -f data/mitdb/$r.atr ] && [ $(stat -f %z data/mitdb/$r.dat 2>/dev/null || echo 0) -eq 1950000 ] && n=$((n+1)); done
echo "$n / 48"
echo "DOWNLOAD DONE" > data/download_status.txt
