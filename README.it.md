# Creare un modello compatibile con Vosk
Questa guida cerca di spiegare come creare un proprio modello compatibile con [vosk](https://alphacephei.com/vosk/), con l’utilizzo di [kaldi](https://kaldi-asr.org/). Premetto che non sono un esperto del progetto kaldi e della tecnologia che c’è dietro allo speech recognition e del machine learning in generale ma, data la difficoltà che io ho avuto nel creare il mio modello, ho voluto comunque condividere una piccola guida su questo.

*Read this in other languages: [English](README.md)*

# Premessa
Prima di cominciare, ti voglio dare alcune dritte su alcuni punti che possono essere fondamentali, e se trascurati possono farti perdere tempo prezioso.

Ovviamete ti consiglio di usare una GPU per l’addestramento, anche una economica (ad esempio io per i miei addestramenti ho usato una Nvidia P620), altrimenti potresti avere dei training lunghi giorni, se non settimane. 
Per fare un buon modello, inoltre, utilizzare molti speaker e molte frasi renderà sempre migliore la qualità del riconoscimento vocale. Se sei da solo e hai bisogno di scaricare qualche dataset da aggiungere al tuo, puoi provare da questo [sito](https://sourceforge.net/projects/cmusphinx/files/Acoustic%20and%20Language%20Models/). Io ho usato questo per avere un dataset da dove cominciare per la lingua italiana e con vari piccoli script in python sono riuscito ad adattarli per kaldi, ma se trovi qualche altro sito ancora più semplice da usare puoi sicuramente utilizzarlo.

Se nel corso della guida andrai incontro a degli errori, ricorda di consultare la sezione [TROUBLESHOTTING](#troubleshooting) infondo alla guida perchè può esserti di aiuto.

# Preparazione
Se dovrai fare l'addestramento con la GPU, scarica e installa cuda prima di andare avanti e controlla la compatibilità tra cuda e le versioni di gcc e g++.

Come prima cosa per iniziare a creare il proprio dataset, bisogna scaricare il progetto kaldi da github con il seguente comando:
```
git clone https://github.com/kaldi-asr/kaldi.git
```
Una volta scaricato, bisogna compilare tutti i programmi che serviranno e ti aiuteranno con la preparazione del dataset e con l'addestramento. Quindi esegui il seguente comando:
```
cd kaldi/tools/; make; cd ../src; ./configure; make
```
Se NON hai intenzione di usare la GPU per l’addestramento, il comando `./configure` deve diventare:
```
./configure --use-cuda=no
```
Poi devi modificare il file `cmd.sh` sotto `kaldi/egs/mini_librispeech/s5` (**che è la directory dove lavorerai fino alla fine della guida**): modica tutti i `queue.pl` con `run.pl`.

Infine devi scaricare anche questo progetto:
```
git clone https://github.com/matteo-39/vosk-build-model.git
```
rendere eseguibili gli script:
```
sudo chmod +x vosk-build-model/*.sh
```
e copiarli nella giusta cartella:
```
cp vosk-build-model/*.sh PATH_TO_KALDI/egs/mini_librispeech/s5/
```

# Creazione Data
Seguendo questa [guida](https://kaldi-asr.org/doc/data_prep.html) ufficiale di kaldi, riuscirai a creare la directory `data/train` con i suoi file necessari per l’addestramento. Per evitare problemi futuri, quando stai creando il file `data/train/text` puoi usare questo tipo di formattazione per l’utterance-id:
```
NOMESPEAKER-0 TESTO
NOMESPEAKER-1 TESTO
NOMESPEAKER-2 TESTO
...
```
Invece per quanto riguarda il file `data/train/wav.scp`, puoi fare in questo modo:
```
NOMESPEAKER-0 PATH_TO_FILE.wav
NOMESPEAKER-1 PATH_TO_FILE.wav
NOMESPEAKER-2 PATH_TO_FILE.wav
...
```
Come avrai letto, i file `data/train/segments`, `data/train/reco2file_and_channel` e `data/train/spk2gender` sono facoltativi, quindi sta a te scegliere se sono necessari per il tuo modello.

> :warning: **WARNING**: I file audio che andrai a registrare o scaricare da internet per il tuo dataset, devono avere un formato simile a: `RIFF (little-endian) data, WAVE audio, Microsoft PCM, 16 bit, mono 16000 Hz`. Questo lo puoi controllare con il comando `file` su linux.
Nel caso contrario potresti avere problemi, ad esempio con il comando `steps/make_mfcc.sh --nj 20 --cmd "$train_cmd" data/train exp/make_mfcc/train $mfccdir`.

Una volta che hai creato tutti i file, puoi controllare se tutto è corretto con i seguenti comandi:
```
utils/validate_data_dir.sh data/train
utils/fix_data_dir.sh data/train (nel caso di errori con il comando precedente)
```
L’ultimo passaggio, ma non meno importante, è quello di eseguire questo comando:
```
cp -R data/train data/test
```

# Creazione Lang
Per la creazione della directory `data/lang` hai bisogno di creare solamente un file, ovvero `data/local/dict/lexicon.txt`. Questo file è composto da ogni singola parola presente nei tuoi audio e dal suo fonema. Per trovare un programma gratuito che permetta di avere il fonema di una parola del dizionario italiano, ma non solo, ho dovuto cercare molto, e ho trovato espeak che con il comando `espeak -q -v it --ipa=3 prova` ti restituisce il fonema, in questo esempio della parola 'prova'.

L'opzione `-q` serve per non riprodurre nessuna voce, `-v` indica la lingua, `--ipa` fa visualizzare il fonema secondo l'International Phonetic Alphabet, mentre l’argomento `3` nell'opzione `--ipa` indica che l’output del fonema sarà spezzettato da degli underscore.

Questo sarà utile poiché nel file `data/local/dict/lexicon.txt` il fonema dovrà avere una forma del tipo:
```
prova p r ˈɔː v a
ciao tʃ ˈaʊ
...
```
Quindi con uno script in python, bash e via dicendo, potrai sostituire gli underscore con uno spazio.

Quando hai finito con `data/local/dict/lexicon.txt`, esegui questi comandi per creare gli altri file sotto `data/local/dict/`:
## nonsilence_phones.txt
```
cut -d ' ' -f 2- lexicon.txt | sed 's/ /\n/g' | sort -u > nonsilence_phones.txt
```
> :information_source: Dopo aver eseguito questo comando controllate, con qualsiasi editor di testo, se la prima riga del file non sia vuota, altrimenti devi eliminarla.
## silence_phones.txt
```
echo -e 'SIL\noov\nSPN' > silence_phones.txt
```
## optional_silence.txt
```
echo 'SIL' > optional_silence.txt
```
Una volta aver creato tutto, importante è poi l’aggiunta di questa riga all’interno di `data/local/dict/lexicon.txt`, che serve per evitare l’errore `undefined symbol <UNK>` (per convenzione la mettiamo all’inizio del file):
```
<UNK> SPN
prova p r ˈɔː v a
ciao tʃ ˈaʊ
...
```
Ora puoi eseguire:
```
utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang data/lang
```
Anche qui puoi controllare se tutto è corretto con i comandi:
```
utils/validate_lang.pl data/lang
utils/validate_dict_dir.pl data/local/dict
```

# Creazione del Language Model
Il language model è composto dal file `data/lang/G.fst` e il file `data/local/tmp/lm.arpa`. Per crearli bisogna eseguire lo script `lm_creation.sh`. Prima
però c’è bisogno che tu crei un altro file: `data/local/corpus.txt`. 
Questo nuovo file deve avere un numero di righe uguali a quanti sono gli audio (ad esempio per 100 audio 100 righe). Nel caso che tu non lo abbia creato precedentemente, lo puoi semplicemente ricavare dal file `data/train/text` con uno script che elimina l’utterance-id. 
Inoltre, per eseguire `lm_creation.sh`, ti serve installare la libreria SRILM. Per installarla scarica il file .tar.gz da questo [sito](http://www.speech.sri.com/projects/srilm/download.html). Una volta scaricato, rinomina il file in modo che non ci sia il numero della versione, quindi devi ritrovarti con il file chiamato in questo modo: `srilm.tar.gz`. Ora prendi il file e mettetilo sotto la cartella `kaldi/tools` e esegui:
```
./install_srilm.sh && ./env.sh
```
Se `env.sh` non viene eseguito, devi renderlo eseguibile con il comando `chmod`.
Ora puoi creare il tuo language model con il seguente comando:
```
./lm_creation.sh
```

# Allignamenti e Training 
Prima di poter far partire l’addestramento vero e proprio, devi completare altre fasi come l'allignamento e training monofonico e così via.
Per fare tutto questo, basta eseguire questo comando:
```
./align_train.sh
```

# Addestramento
Come ultime cose, devi modificare due righe all’interno dello script dell’addestramento con `nano local/chain/tuning/run_tdnn_1j.sh` o qualsiasi altro editor di testo:
```
train_set=train_clean_5
test_sets=dev_clean_2
```
sostituiscilo con:
```
train_set=train
test_sets=test
```
Sempre all’interno di `local/chain/tuning/run_tdnn_1j.sh`, modifica:
```
--use-gpu=yes
```
con:
```
--use-gpu=wait (se NON devi usare la GPU sostituisci “wait” con “no”)
```
e poi esegui anche:
```
sudo nvidia-smi -c 3
```
Il motivo di questo comando e dell’ultima modifica sono citati [qui](https://kaldi-asr.org/doc/cudamatrix.html).
Accertati di aver bisogno di usare la GPU in modalità “wait”. Nel caso tu abbia provato ad avviare l’addestramento con “yes” e poi hai riscontrato un errore tipo `error: core dump`, prova ad usare “wait”.
Dopo esegui anche `nano local/nnet3/run_ivector_common.sh` e modifica le righe:
```
train_set=train_clean_5
test_sets=”dev_clean_2”
```
con:
```
train_set=train
test_sets=”test”
```
Ora esegui l’addestramento:
```
local/chain/tuning/run_tdnn_1j.sh
```

# Ricavare modello
Se l’addestramento non ti ha dato nessun errore, per avere il tuo modello compatibile con vosk puoi iniziare a prendere tutti i file necessari e metterli in una cartella. Questo viene fatto eseguendo:
```
./copy_final_result.sh
```
Come ultima cosa devi organizzare quei file in modo che vosk non abbia problemi. Vedendo dal loro [sito](https://alphacephei.com/vosk/models) ufficiale, infondo alla pagina, c’è una sezione chiamata ‘Model Structure’. Controlla i file che hai nella tua cartella e posizionali in quel modo.
Avrai notato che vosk dice che il file `conf/model.conf` deve essere creato da te perché non è presente dopo l’addestramento. In tutti i miei modelli ho sempre creato quel file con all’interno le seguenti righe:
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
Ora hai il tuo modello perfettamente compatibile con vosk.

# Troubleshooting
- Se, mentre fai il `make` sotto la cartella `src`, ti compare un errore che ti dice ad esempio `questa versione di cuda supporta versioni di gcc <= 7.0`, dopo aver installato la corretta versione, dovrai rifare prima il `make` sotto la cartella `tools`.
- Durante l'esecuzione di `./configure` ti potrebbe comparire un errore in cui ti chiede di scaricare la libreria MKL. Se sei su una distribuzione basata su debian, per scaricarla devi semplicemente eseguire `sudo apt install intel_mkl` . Nell'installazione ti chiederà di sostituire un'altra libreria a 'BLAS and LAPACK'; io non l’ho mai fatto. Se anche essendo su debian non trovi il pacchetto sui tuoi repositori, segui questa [guida](https://www.r-bloggers.com/2018/04/18-adding-intel-mkl-easily-via-a-simple-script/).
- Se quando stai cercando di eseguire il comando `steps/make_mfcc.sh --nj 20 --cmd "$train_cmd" data/train exp/make_mfcc/train $mfccdir` riscontri l'errore `steps/make_mfcc.sh --nj 20 --cmd data/train exp/make_mfcc/train steps/make_mfcc.sh: empty argument to --cmd option` devi solamente sostituire a `$train_cmd` `run.pl`, quindi il comando diventerà: `steps/make_mfcc.sh --nj 20 --cmd "run.pl" data/train exp/make_mfcc/train $mfccdir`
- Se hai riscontrato questo errore `skipped: word WORD not in symbol state`, vuole dire che
all’interno di `data/lang/words.txt` non c’è quella determinata parola. Per risolverlo devi correggere il
file `data/local/dict/lexicon.txt`, perché molto probabilmente non c’è nemmeno lì, e eseguire di nuovo `cut -d ' ' -f 2- lexicon.txt | sed 's/ /\n/g' | sort -u > nonsilence_phones.txt` e `utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang data/lang`
