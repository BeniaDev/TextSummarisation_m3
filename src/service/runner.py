import logging

import typer

from typing import Optional
from pathlib import Path

logging.basicConfig(filename=Path('./logs/app.log'), level=logging.INFO, format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S', filemode="a")

app = typer.Typer()

from luhn_summarizer import LuhnSummarizer

summarizer = LuhnSummarizer()


@app.command()
def summarize(document_path: Optional[Path]):
    """
    API call to model.summarize()
    :param document_path: Article Text for summarization
    """
    with open(document_path, "r") as f:
        doc = "".join(f.readlines())

    summary = summarizer.summarize(doc)

    # saving results
    save_path = Path("./data/summary_results/ru_summary_test.txt")
    with open(save_path, "w") as f:
        f.writelines(summary)

    logging.info(f"Summary for {document_path} has been saved to {save_path} successfully!")



if __name__ == '__main__':
    app()
