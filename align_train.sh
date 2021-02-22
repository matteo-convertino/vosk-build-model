#!/bin/bash

. ./cmd.sh
. ./path.sh

. utils/parse_options.sh

stage=0

if [ $stage -le 0 ]; then
  echo
  echo "==== train a monophone system  ===="
  echo
  steps/train_mono.sh --boost-silence 1.25 --nj 5 --cmd "$train_cmd" \
      data/train data/lang exp/mono

  steps/align_si.sh --boost-silence 1.25 --nj 5 --cmd "$train_cmd" \
    data/train data/lang exp/mono exp/mono_ali_train
fi

if [ $stage -le 1 ]; then
  echo
  echo "==== train a first delta + delta-delta triphone system on all utterances ===="
  echo
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
      2000 10000 data/train data/lang exp/mono_ali_train exp/tri1

  steps/align_si.sh --nj 5 --cmd "$train_cmd" \
      data/train data/lang exp/tri1 exp/tri1_ali_train
fi

if [ $stage -le 2 ]; then
  echo
  echo "==== train an LDA+MLLT system ===="
  echo
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" 2500 15000 \
    data/train data/lang exp/tri1_ali_train exp/tri2b

  steps/align_si.sh  --nj 5 --cmd "$train_cmd" --use-graphs true \
    data/train data/lang exp/tri2b exp/tri2b_ali_train
fi

if [ $stage -le 3 ]; then
  echo
  echo "==== Train tri3b, which is LDA+MLLT+SAT ===="
  echo
  steps/train_sat.sh --cmd "$train_cmd" 2500 15000 \
    data/train data/lang exp/tri2b_ali_train exp/tri3b
fi

if [ $stage -le 4 ]; then
  echo
  echo "==== Now we compute the pronunciation and silence probabilities from training data,"
  echo "and re-create the lang directory. ===="
  echo
  steps/get_prons.sh --cmd "$train_cmd" \
    data/train data/lang exp/tri3b

  # Prevent the lexicon from becoming empty and giving an error. In this way the next command will redo it from scratch.
  mv data/local/dict/lexicon.txt data/local/dict/lexicon_old.txt

  utils/dict_dir_add_pronprobs.sh --max-normalize true \
    data/local/dict \
    exp/tri3b/pron_counts_nowb.txt exp/tri3b/sil_counts_nowb.txt \
    exp/tri3b/pron_bigram_counts_nowb.txt data/local/dict
  utils/prepare_lang.sh data/local/dict \
    "<UNK>" data/local/lang data/lang

  . ./lm_creation.sh


  utils/build_const_arpa_lm.sh \
    data/local/tmp/lm.arpa.gz data/lang data/lang_test_tglarge

  steps/align_fmllr.sh --nj 5 --cmd "$train_cmd" \
    data/train data/lang exp/tri3b exp/tri3b_ali_train
fi

if [ $stage -le 5 ]; then
  echo
  echo "==== Test the tri3b system with the silprobs and pron-probs."
  echo "decode using the tri3b model ===="
  echo
  cp -r data/lang data/lang_test_tgsmall
  cp -r data/lang data/lang_test_tgmed

  utils/mkgraph.sh data/lang_test_tgsmall \
    exp/tri3b exp/tri3b/graph_tgsmall
  for test in test; do
    steps/decode_fmllr.sh --nj 10 --cmd "$decode_cmd" \
                          exp/tri3b/graph_tgsmall data/$test \
                          exp/tri3b/decode_tgsmall_$test
    steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_{tgsmall,tgmed} \
                       data/$test exp/tri3b/decode_{tgsmall,tgmed}_$test
    steps/lmrescore_const_arpa.sh \
      --cmd "$decode_cmd" data/lang_test_{tgsmall,tglarge} \
      data/$test exp/tri3b/decode_{tgsmall,tglarge}_$test
  done
fi

echo
echo "Done"
