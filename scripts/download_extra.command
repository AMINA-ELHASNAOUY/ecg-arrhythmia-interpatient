#!/bin/bash
# Resumable download of two extra PhysioNet databases used by extensions.py:
#   nstdb - MIT-BIH Noise Stress Test Database (record 118 with added noise at 6 levels, ~5 MB)
#   afdb  - MIT-BIH Atrial Fibrillation Database (23 ten-hour recordings with AF, ~650 MB)
cd "$(dirname "$0")/.."
fsize() { if [ -f "$1" ]; then wc -c < "$1" | tr -d ' '; else echo 0; fi; }
get() {  # $1 db  $2 file
  curl -fsS --retry 2 --connect-timeout 15 --max-time 1200 -C - -o "data/$1/$2" "https://physionet-open.s3.amazonaws.com/$1/1.0.0/$2" ||
  curl -fsS --retry 3 --connect-timeout 20 --max-time 1200 -C - -o "data/$1/$2" "https://physionet.org/files/$1/1.0.0/$2"
}
complete() {  # $1 db  $2 record  $3.. annotation extensions
  local db=$1 r=$2; shift 2
  local h=data/$db/$r.hea
  [ -s $h ] && ! head -c 6 $h | grep -q "<" || return 1
  for e in "$@"; do [ -s data/$db/$r.$e ] || return 1; done
  local want=$(awk 'NR==1{n=$2; s=$4} NR==2{f=$2} END{split(f,a,"x"); if (a[1]==212) print int(n*s*3/2); else print n*s*2}' $h)
  [ $(fsize data/$db/$r.dat) -ge $want ]
}
export -f get
fetch() {  # $1 db  $2 "records"  $3 "annotation extensions"
  mkdir -p data/$1
  for round in 1 2 3 4 5; do
    todo=""
    for r in $2; do complete $1 $r $3 || todo="$todo $r"; done
    n=$(echo $todo | wc -w | tr -d ' ')
    echo "$1 round $round: $n records to fetch"
    [ $n -eq 0 ] && return 0
    for r in $todo; do echo $r; done | xargs -P 6 -I{} bash -c "get $1 {}.hea; for e in $3; do get $1 {}.\$e; done; get $1 {}.dat && echo 'ok $1 {}'"
  done
}
fetch nstdb "118e24 118e18 118e12 118e06 118e00 118e_6" "atr"   # 118: a DS1 hold-out patient, never trained on
# 00735 and 03665 have no signal files, so they are not included
fetch afdb "04015 04043 04048 04126 04746 04908 04936 05091 05121 05261 06426 06453 06995 07162 07859 07879 07910 08215 08219 08378 08405 08434 08455" "atr qrs"
echo "EXTRA DONE"
