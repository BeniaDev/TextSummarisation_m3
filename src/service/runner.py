import logging

import typer

from typing import Optional
from pathlib import Path

logging.basicConfig(filename='../logs/app.log', level=logging.INFO, format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S', filemode="a")

app = typer.Typer()


@app.command()
def train():
    """
    API call for model.train()
    :param dataset: path to train dataset
    :return: None
    """
    pass


@app.command()
def predict():
    """
    API call to model.recommend()
    :param user_id: user id in System
    :param M: count of recommend films
    :return: (movies_id_list, predicted_ratings_list)
    """
    pass



@app.command()
def reload():
    """
    API call to model.warmup(). Just reload the model from /app/model/
    :return: None
    """
    pass


if __name__ == '__main__':
    app()
