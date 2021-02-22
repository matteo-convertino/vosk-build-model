#!/bin/bash

. ./cmd.sh
. ./path.sh

. utils/parse_options.sh

echo "===== LANGUAGE MODEL CREATION ====="
echo "===== MAKING lm.arpa ====="
echo

loc=`which ngram-count`;

if [ -z $loc ]; then
  if uname -a | grep 64 >/dev/null; then
    sdir=$KALDI_ROOT/tools/srilm/bin/i686-m64
  else
    sdir=$KALDI_ROOT/tools/srilm/bin/i686
  fi
  if [ -f $sdir/ngram-count ]; then
    echo "Using SRILM language modelling tool from $sdir"
    export PATH=$PATH:$sdir
  else
    echo "SRILM toolkit is probably not installed.
          Instructions: tools/install_srilm.sh"
    exit 1
   fi
fi

local=data/local
lang=data/lang
mkdir $local/tmp
ngram-count -order $lm_order -write-vocab $local/tmp/vocab-full.txt -wbdiscount -text $local/corpus.txt -lm $local/tmp/lm.arpa

echo
echo "==== Making G.fst ===="
echo

arpa2fst --max-arpa-warnings=-1 --disambig-symbol=#0 --read-symbol-table=$lang/words.txt $local/tmp/lm.arpa $lang/G.fst
tar -czvf data/local/tmp/lm.arpa.gz data/local/tmp/lm.arpa

echo "Done"
