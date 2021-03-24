# Build model for Vosk
This guide tries to explain how to create your own compatible model with [vosk](https://alphacephei.com/vosk/), with the use of [kaldi](https://kaldi-asr.org/). I state that I am not an expert on the kaldi project and on the technology behind speech recognition and deep learning in general but, given the difficulty I had in creating my model, I still wanted to share a little guide about this. I also apologize if the English level of this guide will be bad.

*Read this in other languages: [Italian](README.it.md)*

# Premise
Before starting, I want to give you some tips on some points that can be crucial, and if ignored can waste your precious time.

Obviously I suggest you to use a GPU for training, even a cheap one (for example I used a Nvidia P620 for my training), otherwise you could have trainings that last for days or weeks.
To make a good model, also, using many speakers and many phrases will always make the voice recognition quality better. If you are on your own and you need to download some datasets to add to yours, you can try from this [site](https://sourceforge.net/projects/cmusphinx/files/Acoustic%20and%20Language%20Models/). I used this to get a dataset where to start for the italian language and with various small scrips in python I was able to adapt it for kaldi, but if you find some other site even easier to use you can definitely use it.

If you encounter any errors during the course of this guide, remember to consult the [TROUBLESHOOTING](#troubleshooting) section at the end of the guide because it may help you.

# Preparation 
If you're going to be training with the GPU, download and install cuda before you go ahead and check compatibility between cuda and versions of gcc and g++.

First of all to start creating your dataset, you need to download the kaldi project from github with the following command:
```
git clone https://github.com/kaldi-asr/kaldi.git
```
Once downloaded, you have to compile all the programs that you will need and will help you with dataset preparation and training. Then run the following command:
```
cd kaldi/tools/; make; cd ../src; ./configure; make
```
If you are NOT going to use the GPU for training, the command `./configure` must become:
```
./configure --use-cuda=no
```
Then you have to edit the file `cmd.sh` under `kaldi/egs/mini_librispeech/s5` (**which is the directory where you will work until the end of the guide**): change all `queue.pl` to `run.pl`.

Finally you have to download this project too:
```
git clone https://github.com/matteo-39/vosk-build-model.git
```
make the scripts executable:
```
sudo chmod +x vosk-build-model/*.sh
```
and copy them to the right directory:
```
cp vosk-build-model/*.sh PATH_TO_KALDI/egs/mini_librispeech/s5/
```

# Data Creation
By following this official kaldi [guide](https://kaldi-asr.org/doc/data_prep.html), you will be able to create the `data/train` directory with its necessary training files. To avoid future problems, when you are creating the `data/train/text` file you can use this kind of formatting for the utterance-id:
```
SPEAKERNAME-0 TEXT
SPEAKERNAME-1 TEXT
SPEAKERNAME-2 TEXT
...
```
As for the `data/train/wav.scp` file, you can do this:
```
SPEAKERNAME-0 PATH_TO_FILE.wav
SPEAKERNAME-1 PATH_TO_FILE.wav
SPEAKERNAME-2 PATH_TO_FILE.wav
...
```
As you may have read, the files `data/train/segments`,` data/train/reco2file_and_channel` and `data/train/spk2gender` are optional, so it's up to you to choose if they are needed for your model. 

> :warning: **WARNING**: The audio files that you are going to record or download from the internet for your dataset must have a format similar to: `RIFF (little-endian) data, WAVE audio, Microsoft PCM, 16 bit, mono 16000 Hz`. You can check this with the `file` command on linux.
Otherwise you may have problems, for example with the command `steps/make_mfcc.sh --nj 20 --cmd "$train_cmd" data/train exp/make_mfcc/train $mfccdir`.

Once you have created all the files, you can check if everything is correct with the following commands:
```
utils/validate_data_dir.sh data/train
utils/fix_data_dir.sh data/train (in case of errors with the previous command)
```

# Lang Creation
To create the `data/lang` directory you only need to create one file, namely` data/local/dict/lexicon.txt`. This file consists of every single word in your audio and its phoneme.
To find a free program that allows you to have the phoneme of a word of the Italian dictionary, but not only, I had to search a lot, and I found espeak that with the command `espeak -q -v it --ipa=3 test` returns the phoneme, in this example of the word 'test'.

The `-q` option is for not playing any voices, `-v` indicates the language, `--ipa` displays the phoneme according to the International Phonetic Alphabet, and the `3` argument in the `--ipa` option indicates that the output of the phoneme will be broken up by underscores.

This will be useful since in the file `data/local/dict/lexicon.txt` the phoneme should have a form like:
```
test t ˈɛ s t
hi h ˈaɪ
...
```
So with a script in python, bash and so on, you can replace the underscores with a space.

When you are done with `data/local/dict/lexicon.txt`, run these commands to create the other files under `data/local/dict/`:
## nonsilence_phones.txt
```
cut -d ' ' -f 2- lexicon.txt | sed 's/ /\n/g' | sort -u > nonsilence_phones.txt
```
> :information_source: After running this command check, with any text editor, if the first line of the file is not empty, otherwise you have to delete it.
## silence_phones.txt
```
echo -e 'SIL\noov\nSPN' > silence_phones.txt
```
## optional_silence.txt
```
echo 'SIL' > optional_silence.txt
```
Once everything is created, it is important to add this line inside `data/local/dict/lexicon.txt`, which is used to avoid the error `undefined symbol <UNK>` (by convention we put it at the beginning of the file):
```
<UNK> SPN
test t ˈɛ s t
hi h ˈaɪ
...
```
Now you can run: 
```
utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang data/lang
```
Here too you can check if everything is correct with the commands:
```
utils/validate_lang.pl data/lang
utils/validate_dict_dir.pl data/local/dict
```

# Language Model Creation
The language model consists of the `data/lang/G.fst` file and the` data/local/tmp/lm.arpa` file. To create them you need to run the `lm_creation.sh` script. First, however, you need to create another file: `data/local/corpus.txt`.
This new file must have a number of lines equal to the number of audios (for example for 100 audios 100 lines). In case you haven't created it previously, you can simply get it from the `data/train/text` file with a script that removes the utterance-id. 
Also, to run `lm_creation.sh`, you need to install the SRILM library. To install it, download the .tar.gz file from this [site](http://www.speech.sri.com/projects/srilm/download.html). Once downloaded, rename the file so that there is no version number, so you have to end up with the file named like this: `srilm.tar.gz`. Now take the file and put it under the `kaldi/tools` folder and run it:
```
./install_srilm.sh && ./env.sh
```
If `env.sh` does not run, you must make it executable with the `chmod` command.
Now you can create your language model with the following command:
```
./lm_creation.sh
```

# Alignaments and Training   
Before you can start the actual training, you have to complete other steps such as alignment and monophonic training and so on.
To do all this, just run this command:
```
./align_train.sh
```

# Training
As last things, you have to edit two lines inside the training script with `nano local/chain/tuning/run_tdnn_1j.sh` or any other text editor:
```
train_set=train_clean_5
test_sets=dev_clean_2
```
replace it with:
```
train_set=train
test_sets=test
```
Still within `local/chain/tuning/run_tdnn_1j.sh`, edit:
```
--use-gpu=yes
```
with:
```
--use-gpu=wait (if you do NOT have to use the GPU replace "wait" with "no".)
```
and then also run:
```
sudo nvidia-smi -c 3
```
The reason for this command and the last change are cited [here](https://kaldi-asr.org/doc/cudamatrix.html).
Make sure you need to use the GPU in "wait" mode. In case you tried to start the training with "yes" and then got an error like `error: core dump`, try using "wait".
After that also run `nano local/nnet3/run_ivector_common.sh` and edit the lines:
```
train_set=train_clean_5
test_sets=”dev_clean_2”
```
with:
```
train_set=train
test_sets=”test”
```
Now run the training:
```
local/chain/tuning/run_tdnn_1j.sh
```

# Get model
If the training didn't give you any error, to have your model compatible with vosk you can start by taking all the necessary files and put them in a folder. This is done by running:
```
./copy_final_result.sh
```
As a last thing you need to organize those files so that vosk doesn't have any problems. Seeing from their official [site](https://alphacephei.com/vosk/models), at the bottom of the page, there is a section called 'Model Structure'. Check the files you have in your folder and place them that way.
You may have noticed that vosk says that the `conf/model.conf` file must be created by you because it is not present after training. In all my models I have always created that file with the following lines inside:
```
--min-active=200
--max-active=3000
--beam=10.0
--lattice-beam=2.0
--acoustic-scale=1.0
--frame-subsampling-factor=3
--endpoint.silence-phones=1:2:3:4:5:6:7:8:9:10
--endpoint.rule2.min-trailing-silence=0.5
--endpoint.rule3.min-trailing-silence=1.0
--endpoint.rule4.min-trailing-silence=2.0
```
Now you have your model perfectly compatible with vosk.

# Troubleshooting
- If, while running the `make` under the `src` folder, you get an error saying for example `this version of cuda supports gcc versions <= 7.0`; after installing the correct version, you will have to run the `make` under the `tools` folder first.
- When running `./configure` you may get an error asking you to download the MKL library. If you are on a debian based distribution, to download it you simply run `sudo apt install intel-mkl` . In the installation it will ask you to replace another library for 'BLAS and LAPACK'; I never did that. If even being on debian you don't find the package on your repositories, follow this [guide](https://www.r-bloggers.com/2018/04/18-adding-intel-mkl-easily-via-a-simple-script/).
- If when you are trying to execute the command `steps/make_mfcc.sh --nj 20 --cmd "$train_cmd" data/train exp/make_mfcc/train $mfccdir` you get the error `steps/make_mfcc.sh --nj 20 --cmd data/train exp/make_mfcc/train steps/make_mfcc. sh: empty argument to --cmd option` you just have to replace `$train_cmd` with `run.pl`, so the command will become: `steps/make_mfcc.sh --nj 20 --cmd "run.pl" data/train exp/make_mfcc/train $mfccdir`.
- If you got this error `skipped: word WORD not in symbol state`, it means that
within `data/lang/words.txt` there is not that particular word. To solve it you have to correct the
file `data/local/dict/lexicon.txt`, because most likely it's not there either, and run again `cut -d ' ' -f 2- lexicon.txt | sed 's/ /\n/g' | sort -u > nonsilence_phones.txt` and `utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang data/lang`
-  It may happen that the training crashes during iterations without an error and if you try to run `nvidia-smi` it will crash too. To fix this bug you need to run `sudo nvidia-smi -pm 1` before training.
