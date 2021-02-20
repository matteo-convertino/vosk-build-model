# Creare un modello compatibile con Vosk
Questa guida cerca di spiegare come creare un proprio modello compatibile con [vosk](https://alphacephei.com/vosk/), con l’utilizzo di [kaldi](https://kaldi-asr.org/). Premetto che non sono un esperto del progetto kaldi e della tecnologia che c’è dietro allo speech recognition e del machine learning in generale ma, data la difficoltà che io ho avuto nel creare il mio modello, ho voluto comunque condivere una piccola guida su questo.

# Premessa
Prima di cominciare, ti voglio dare alcune dritte su alcuni punti che possono essere fondamentali, e se trascurati possono farti perdere tempo prezioso.

Ovviamete ti consiglio di usare una gpu per l’addestramento, anche una economica (ad esempio io per i miei addestramenti ho usato una Nvidia P620), altrimenti potresti avere dei training lunghi giorni, se non settimane. 
Per fare un buon modello, inoltre, utilizzare molti speaker e molte frasi renderà sempre migliore la qualità del riconoscimento vocale. Se sei da solo e hai bisogno di scaricare qualche dataset da aggiungere al tuo, puoi provare da questo [sito](https://sourceforge.net/projects/cmusphinx/files/Acoustic%20and%20Language%20Models/). Io ho usato questo per avere un dataset da dove cominciare per la lingua italiana e con vari piccoli script in python sono riuscito ad adattarli per kaldi, ma sei trovi qualche altro sito ancora più semplice da usare puoi sicuramente farlo.

Se nel corso della guida andrai incontro a degli errori, ricorda di consultare la sezione [TROUBLESHOTTING](#troubleshooting) infondo alla guida.

# Preparazione
Se dovrai fare l'addestramento con la gpu, scarica e installa cuda prima di andare avanti e controlla la compatibilità tra cuda e le versioni di gcc e g++. Se non lo fai, durante i prossimi comandi, potresti andare incontro a delle interruzioni improvvise, senza un errore ben preciso.

Come prima cosa per iniziare a creare il proprio dataset, bisogna scaricare il progetto di kaldi da github con il seguente comando:
```
git clone https://github.com/kaldi-asr/kaldi.git
```
Una volta scaricato bisogna compilare tutti i programmi che serviranno e ti aiuteranno con la preparazione del dataset e con l'addestramento. Quindi esegui il seguente comando:
```
cd kaldi/tools/; make; cd ../src; ./configure; make
```
Se NON hai intenzione di usare la gpu per l’addestramento, il comando ./configure deve diventare:
```
./configure --use-cuda=no
```
Infine devi modificare il file cmd.sh sotto kaldi/egs/mini_librispeech/s5 (**che è la directory dove lavorerai fino alla fine della guida**): modica tutti i `queue.pl` con `run.pl`.

# Creazione Data
Seguendo questa [guida](https://kaldi-asr.org/doc/data_prep.html) ufficiale di kaldi, riuscirai a creare i file necessari per l’addestramento. Per evitare problemi futuri, quando stai creando il file `data/train/text` puoi usare questo tipo di formattazione per l’utterance-id:
```
NOMESPEAKER-0 STRINGA
NOMESPEAKER-1 STRINGA
NOMESPEAKER-2 STRINGA
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

> :warning: **WARNING**: I file audio che andrai a registrare o scaricare da internet per il tuo dataset, devono avere un formato simile a: `RIFF (little-endian) data, WAVE audio, Microsoft PCM, 16 bit, mono 16000 Hz`. Questo lo puoi controllare con il comando file su linux.
Nel caso contrario potresti avere problemi, ad esempio col comando `steps/make_mfcc.sh --nj 20 --cmd "$train_cmd" data/train exp/make_mfcc/train $mfccdir`.

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
Per la creazione della directory `data/lang` hai bisogno di creare solamente un file, ovvero `data/local/dict/lexicon.txt`. Questo file è composto da ogni singola parola presente nei tuoi audio e dal suo fonema. Per trovare un programma gratuito che permetta di avere il fonema di una parola del dizionario italiano ho dovuto cercare molto, e ho trovato espeak che con il comando `espeak -q -v it --ipa=3 prova` ti restituisce il fonema, in questo esempio della parola 'prova'.

L'opzione `-q` serve per non riprodurre nessuna voce, `-v` indica la lingua, `--ipa` fa visualizzare il fonema secondo l'International Phonetic Alphabet, mentre l’argomento `3` nell'opzione `--ipa` indica che l’output del fonema sarà spezzettato da degli underscore.

Questo sarà utile poiché nel file `lexicon.txt` il fonema dovrà avere una forma del tipo:
```
prova p r ˈɔː v a
ciao tʃ ˈaʊ
...
```
Quindi con uno script in python, bash ecc. potrai sostituire gli underscore con uno spazio.

Quando hai finito con `lexicon.txt`, esegui questi comandi per creare gli altri file sotto `data/local/dict/`:
## nonsilence_phones.txt:
```
cut -d ' ' -f 2- lexicon.txt | sed 's/ /\n/g' | sort -u > nonsilence_phones.txt
```
> :information_source: Dopo aver eseguito questo comando, con qualsiasi editor di testo, elimina la prima riga del file perché è vuota.
## silence_phones.txt:
```
echo -e 'SIL\noov\nSPN' > silence_phones.txt
```
## optional_silence.txt:
```
echo 'SIL' > optional_silence.txt
```
Una volta aver creato tutto, importante è poi l’aggiunta di questa riga all’interno di `lexicon.txt` una volta creato tutto (per convenzione la mettiamo all’inizio del file):


















