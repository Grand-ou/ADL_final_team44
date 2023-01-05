# download data
if [[ ! -d "data" ]]; then
    wget -O data.zip https://www.dropbox.com/s/a7y1592mpg4miyp/data.zip?dl=0
    unzip data.zip
    rm data.zip
fi

python seen.py "${1}"

