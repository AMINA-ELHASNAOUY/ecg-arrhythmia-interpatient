#!/bin/bash
# Extensions (after run_all.command and download_extra.command):
#   real-time CNN (5 seeds, past beats only) -> noise stress test, more AF patients, real-time comparison
cd "$(dirname "$0")/.."
source .venv/bin/activate
fsize() { if [ -f "$1" ]; then wc -c < "$1" | tr -d ' '; else echo 0; fi; }
complete() {  # $1 db  $2 record: .dat has the size its header promises
  local h=data/$1/$2.hea
  [ -s $h ] || return 1
  local want=$(awk 'NR==1{n=$2; s=$4} NR==2{f=$2} END{split(f,a,"x"); if (a[1]==212) print int(n*s*3/2); else print n*s*2}' $h)
  [ $(fsize data/$1/$2.dat) -ge $want ]
}
{
  python -u train.py --realtime --seeds 5 || exit 1
  for i in $(seq 1 120); do              # wait (up to 1 hour) until the AF database download is complete
    missing=0
    for r in 04015 04043 04048 04126 04746 04908 04936 05091 05121 05261 06426 06453 06995 07162 07859 07879 07910 08215 08219 08378 08405 08434 08455; do
      complete afdb $r && [ -s data/afdb/$r.qrs ] || missing=$((missing+1))
    done
    [ $missing -eq 0 ] && break
    echo "waiting for the AF database download: $missing recordings left"; sleep 30
  done
  python -u extensions.py
  echo "exit code $?"
} 2>&1 | tee results/extensions_log.txt
