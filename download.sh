rm -rf data
mkdir -p data
cd data
wget https://raw.githubusercontent.com/espeak-ng/espeak-ng/master/dictsource/extra/ru_listx -O espeak.txt
wget https://www.dropbox.com/s/4stea3h10vbj3pc/ru_custom.txt
wget https://www.dropbox.com/s/o5v3bcc5e3lvewq/zaliznyak.txt
wget https://dumps.wikimedia.org/ruwiktionary/20221201/ruwiktionary-20221201-pages-articles.xml.bz2
bzip2 -d ruwiktionary-20221201-pages-articles.xml.bz2
