from pathlib import Path
from nltk.corpus import stopwords
from src.data_loader import DataLoader
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import click
import pickle
import logging
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]

LSA_K = [30, 50, 100, 200, 300, 500]


@click.command()
@click.option('--log', type=str, default="INFO", help="Logging level.")
def main(log):
    logger = logging.getLogger(__name__)
    logger.setLevel(log.upper())

    data_path = Path(PROJECT_DIR / "data" / "raw" / "loader.pickle")

    if not data_path.exists():
        dl = DataLoader()
        pickle.dump(dl, open(data_path, "wb"))
    else:
        dl = pickle.load(open(data_path, "rb"))

    data = {'reuters': dl.reuters, 'newsgroups': dl.newsgroups, 'imdb': dl.imdb}

    vader = SentimentIntensityAnalyzer()

    for k, v in data.items():
        logger.info("====={0}=====".format(k))
        processed = {}

        train = v['train']
        test = v['test']

        processed['train'] = train
        processed['test'] = test

        # ---------------------------------------------------------
        # -======================== TF-IDF =======================-
        # ---------------------------------------------------------
        tfidf = TfidfVectorizer(
            stop_words=stopwords.words('english'),
            max_df=0.6,
            min_df=2
        )

        logger.info("Training TF-IDF.")

        tfidf.fit(train['clean'])

        logger.info("Applying TF-IDF on train split.")
        processed['tf_train'] = tfidf.transform(train['clean'])

        logger.info("Applying TF-IDF on test split.")
        processed['tf_test'] = tfidf.transform(test['clean'])

        dump_path = Path(
            PROJECT_DIR / "models" / "tfidf" / '{}.pickle'.format(k)
        )

        logger.info("Save TF-IDF model.")
        with open(dump_path, "wb") as f:
            pickle.dump(tfidf, f)

        dump_path = Path(
            PROJECT_DIR / "models" / "lsa" / '{0}_{1}.pickle'
        )

        # ---------------------------------------------------------
        # -========================= LSA =========================-
        # ---------------------------------------------------------
        logger.info("Processing Truncated Singular Vector Decomposition.")
        for rank in LSA_K:
            lsa = TruncatedSVD(rank, n_iter=20, random_state=42).fit(
                processed['tf_train']
            )
            logger.info("Train : LSA-{0} -DONE-".format(rank))

            processed['train_lsa_{0}'.format(rank)] = lsa.transform(
                processed['tf_train']
            )

            logger.info('Train split transformed.')

            processed['test_lsa_{0}'.format(rank)] = lsa.transform(
                processed['tf_test']
            )

            logger.info('Test split transformed.')

            with open(str(dump_path).format(k, rank), "wb") as f:
                pickle.dump(lsa, f)

            logger.info("Model saved.")

        # ---------------------------------------------------------
        # -======================== VADER ========================-
        # ---------------------------------------------------------
        logger.info("Processing VADER.")
        processed['train_vader'] = process_vader(train['raw'], vader)

        logger.info("VADER on train split -Done-")

        processed['test_vader'] = process_vader(test['raw'], vader)

        logger.info("VADER on test split -Done-")

        # ---------------------------------------------------------
        # -======================== DUMP =========================-
        # ---------------------------------------------------------
        dump_path = Path(
            PROJECT_DIR / "data" / "processed" / '{0}.pickle'.format(k)
        )

        with open(dump_path, "wb") as f:
            pickle.dump(processed, f)

    logger.info("Build features -Done-")


def process_vader(data, vader):
    x = {
        'neg': [],
        'neu': [],
        'pos': [],
        'compound': []
    }

    for t in data:
        sentiment = vader.polarity_scores(t)

        for ke in x.keys():
            x[ke].append(sentiment[ke])

    return pd.DataFrame(x)


if __name__ == "__main__":
    frmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=frmt)

    # Create repositories for saving models
    Path(PROJECT_DIR / "models" / "tfidf").mkdir(parents=True, exist_ok=True)
    Path(PROJECT_DIR / "models" / "lsa").mkdir(parents=True, exist_ok=True)

    # Create repository to save prepared data
    Path(PROJECT_DIR / "data" / "raw").mkdir(parents=True, exist_ok=True)
    Path(PROJECT_DIR / "data" / "processed").mkdir(parents=True, exist_ok=True)

    main()
