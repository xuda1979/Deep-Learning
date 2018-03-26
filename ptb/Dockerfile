FROM tensorflow/tensorflow:1.6.0-gpu-py3
WORKDIR ~/ptb
COPY . .

CMD [ "python", "./ptb_word_lm.py", "--data_path=./simple-examples/data/", "--model=large","--logdir=./log"]
