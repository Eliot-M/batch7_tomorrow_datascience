FROM python:3.6
WORKDIR /inversecooking

COPY . /inversecooking
RUN pip install -r ml-requirements.txt


# RUN wget "http://data.csail.mit.edu/im2recipe/det_ingrs.json" \ 
#          "http://data.csail.mit.edu/im2recipe/recipe1M_layers.tar.gz" \
#          -P datasets/

RUN wgset -nc https://dl.fbaipublicfiles.com/inversecooking/ingr_vocab.pkl \
    https://dl.fbaipublicfiles.com/inversecooking/instr_vocab.pkl \
    https://dl.fbaipublicfiles.com/inversecooking/modelbest.ckpt \
    -P src/data/

VOLUME data

CMD ["python", "src/inversecooking-api.py"]