#!/bin/bash
# Resumable download of the St Petersburg INCART 12-lead Arrhythmia Database (~800 MB).
cd "$(dirname "$0")/.."
fsize() { if [ -f "$1" ]; then wc -c < "$1" | tr -d ' '; else echo 0; fi; }   # portable file size (macOS + Linux)
mkdir -p data/incart
BASE=https://physionet-open.s3.amazonaws.com/incartdb/1.0.0
ALT=https://physionet.org/files/incartdb/1.0.0
get() { curl -fsS --retry 2 --connect-timeout 15 --max-time 900 -C - -o "$2" "$BASE/$1" || curl -fsS --retry 3 --connect-timeout 20 --max-time 900 -C - -o "$2" "$ALT/$1"; }
[ -s data/incart/RECORDS ] || get RECORDS data/incart/RECORDS
complete() {  # $1 = record: header present, annotations present, .dat has the size the header promises
  local h=data/incart/$1.hea
  [ -s $h ] && ! head -c 6 $h | grep -q "<" && [ -s data/incart/$1.atr ] || return 1
  local want=$(head -1 $h | awk '{print $2*$4*2}')
  [ -f data/incart/$1.dat ] && [ $(fsize data/incart/$1.dat) -eq $want ]
}
export -f get; export BASE ALT
for round in 1 2 3 4 5 6; do
  todo=""
  for r in $(cat data/incart/RECORDS); do complete $r || todo="$todo $r"; done
  n=$(echo $todo | wc -w | tr -d ' ')
  echo "round $round: $n records to fetch"
  [ $n -eq 0 ] && break
  for r in $todo; do echo $r; done | xargs -P 6 -I{} bash -c 'get {}.hea data/incart/{}.hea; get {}.atr data/incart/{}.atr; get {}.dat data/incart/{}.dat && echo "ok {}"'
done
echo "INCART DONE"
