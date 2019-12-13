python3 -m spacy download en_core_web_sm
wget https://www.dropbox.com/s/5pfajl0dmnpy3jj/model_final.pkl?dl=0 -O best_model.pkl
wget https://www.dropbox.com/s/2yb6iwxfm4nb3v8/W2V?dl=0 -O W2V
python3 hw5_test.py $1 $2

