cd ../../../
mkdir afhq
cd afhq

URL=https://www.dropbox.com/s/t9l9o3vsx2jai3z/afhq.zip?dl=0
ZIP_FILE=./data/afhq.zip
mkdir -p ./data
wget -N $URL -O $ZIP_FILE
unzip $ZIP_FILE -d ./data
rm $ZIP_FILE