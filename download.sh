mkdir -p data
cd data
wget https://www.dropbox.com/s/4stea3h10vbj3pc/ru_custom.txt
wget https://www.dropbox.com/s/o5v3bcc5e3lvewq/zaliznyak.txt
wget https://dumps.wikimedia.org/ruwiktionary/20190501/ruwiktionary-20190501-pages-articles.xml.bz2
bzip2 -d ruwiktionary-20190501-pages-articles.xml.bz2