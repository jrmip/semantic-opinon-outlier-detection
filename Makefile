PYTHON = python3

requirements:
	$(PYTHON) -m pip install pip setuptools wheel
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) src/utils/setup_nltk.py
	$(PYTHON) -m textblob.download_corpora

## Make Dataset
data:
	$(PYTHON) src/data/make_dataset.py data/raw data/processed

features:
	$(PYTHON)
