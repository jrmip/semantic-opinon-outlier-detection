from pathlib import Path

from src.utils.tools import run_exp

import click
import pickle
import logging

PROJECT_DIR = Path(__file__).resolve().parents[1]

C_RATES = [0.01, 0.05, 0.10, 0.15, 0.20]


@click.command()
@click.option('--log', type=str, default="INFO", help="Logging level.")
@click.option('--total_iter', type=int, default=5, help="Number of run.")
def main(log, total_iter):
    logger = logging.getLogger(__name__)
    logger.setLevel(log.upper())

    # ---------------------------------------------------------
    # -======================= REUTERS =======================-
    # ---------------------------------------------------------
    logger.info("Starting experiment on : {0}".format('REUTERS'))

    path = Path(PROJECT_DIR / "data" / "processed" / "reuters.pickle")

    out = Path(PROJECT_DIR / "results" / "{0}" / "{0}_{1}_cont_{2}.pickle")

    with open(path, "rb") as f:
        reuters = pickle.load(f)

    Path(PROJECT_DIR / "results" / "reuters").mkdir(parents=True, exist_ok=True)

    for i in range(total_iter):
        logger.info("================== RUN {0}/{1} ==================".format(
            i, total_iter
        ))
        for c in C_RATES:
            logger.info(
                "--------------- Contamination : {} ---------------".format(c)
            )

            res = run_exp(reuters, 'reuters', c)

            with open(str(out).format("reuters", i, c), "wb") as f:
                pickle.dump(res, f)

    # ---------------------------------------------------------
    # -===================== NEWSGROUPS ======================-
    # ---------------------------------------------------------
    logger.info("Starting experiment on : {0}".format('NEWSGROUPS'))

    path = Path(PROJECT_DIR / "data" / "processed" / "newsgroups.pickle")

    with open(path, "rb") as f:
        newsgroups = pickle.load(f)

    Path(PROJECT_DIR / "results" / "newsgroups").mkdir(
        parents=True, exist_ok=True
    )

    for i in range(total_iter):
        logger.info("================== RUN {0}/{1} ==================".format(
            i, total_iter
        ))
        for c in C_RATES:
            logger.info(
                "------------ Contamination : {} -----------".format(c)
            )

            res = run_exp(newsgroups, 'newsgroups', c)

            with open(str(out).format("newsgroups", i, c), "wb") as f:
                pickle.dump(res, f)


if __name__ == "__main__":
    frmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=frmt)

    main()
