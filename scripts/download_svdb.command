#!/bin/bash
# Resumable download of the MIT-BIH Supraventricular Arrhythmia Database (external test set, ~40 MB).
cd "$(dirname "$0")/.."
fsize() { if [ -f "$1" ]; then wc -c < "$1" | tr -d ' '; else echo 0; fi; }   # portable file size (macOS + Linux)
mkdir -p data/svdb
# AWS Open Data mirror of PhysioNet (the main site rate-limits bulk downloads)
BASE=https://physionet-open.s3.amazonaws.com/svdb/1.0.0
ALT=https://physionet.org/files/svdb/1.0.0
[ -s data/svdb/RECORDS ] || for i in 1 2 3 4 5; do curl -fsS --max-time 60 -o data/svdb/RECORDS $BASE/RECORDS || curl -fsS --max-time 60 -o data/svdb/RECORDS $ALT/RECORDS; [ -s data/svdb/RECORDS ] && break; done
for round in 1 2 3 4 5; do
  for r in $(cat data/svdb/RECORDS); do
    for ext in hea atr dat; do
      f=data/svdb/$r.$ext
      if [ -f $f ] && head -c 6 $f | grep -q "<html"; then rm -f $f; fi
      if [ "$ext" = dat ] && [ -f $f ] && [ $(fsize $f) -ne 691200 ]; then rm -f $f; fi
      [ -s $f ] && [ ! -f $f.part ] && continue
      echo "$ext $r"
    done
  done | xargs -P 8 -n 2 sh -c 'touch data/svdb/$1.$0.part; (curl -fsS --retry 2 --connect-timeout 15 --max-time 120 -o data/svdb/$1.$0 '"$BASE"'/$1.$0 || curl -fsS --retry 3 --connect-timeout 20 --max-time 300 -o data/svdb/$1.$0 '"$ALT"'/$1.$0) && rm -f data/svdb/$1.$0.part && echo "ok $1.$0"'
  missing=0
  for r in $(cat data/svdb/RECORDS); do for ext in hea atr dat; do [ -s data/svdb/$r.$ext ] && [ ! -f data/svdb/$r.$ext.part ] || missing=$((missing+1)); done; done
  echo "round $round: $missing files missing"
  [ $missing -eq 0 ] && break
done
echo "SVDB DONE ($(wc -l < data/svdb/RECORDS) records)"
