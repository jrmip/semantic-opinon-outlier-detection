PYTHON = python3

requirements:
	$(PYTHON) -m pip install pip setuptools wheel
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) src/utils/setup_nltk.py
	$(PYTHON) -m textblob.download_corpora

## Make dataset and prepare features for experiment
features:
	$(PYTHON) src/build_features.py --log=info

experiment:
	$(PYTHON) src/run.py --log=info --total_iter=5

clean:
	rm -R data
	rm -R models
	rm -R SOFOD.egg-info
