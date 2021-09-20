import logging
import pandas as pd

from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.iforest import IForest

from sklearn.metrics import average_precision_score, roc_curve, auc

from src.utils.prepare_splits import prepare_reuters, prepare_newsgroups
from src.build_features import LSA_K


def run_exp(data, dataset, contamination, split_size=-1):
    logger = logging.getLogger(__name__)

    # ---------------------------------------------------------
    # -======================== TF-IDF =======================-
    # ---------------------------------------------------------
    train = pd.DataFrame(data['tf_train'].toarray())
    train['label'] = data['train']['label']
    train['clean'] = data['train']['clean']

    test = pd.DataFrame(data['tf_test'].toarray())
    test['label'] = data['test']['label']
    test['clean'] = data['test']['clean']

    logger.info("=====TF-IDF=====")

    if dataset == 'reuters':
        train = prepare_reuters(train, split_size, contamination)
        test = prepare_reuters(test, split_size, contamination)
    elif dataset == 'newsgroups':
        train = prepare_newsgroups(train, split_size, contamination)
        test = prepare_newsgroups(test, split_size, contamination)

    tfidf_tr, tfidf_te = run_models(train, test, contamination=contamination)

    logger.info("Experiment on TF-IDF -Done-")

    # ---------------------------------------------------------
    # -========================= LSA =========================-
    # ---------------------------------------------------------
    lsa_tr = {}
    lsa_te = {}

    for rank in LSA_K:
        logger.info("=====LSA-{0}=====".format(rank))

        train = pd.DataFrame(data['train_lsa_{}'.format(rank)])
        train['label'] = data['train']['label']
        train['clean'] = data['train']['clean']

        test = pd.DataFrame(data['test_lsa_{}'.format(rank)])
        test['label'] = data['test']['label']
        test['clean'] = data['test']['clean']

        if dataset == 'reuters':
            train = prepare_reuters(train, split_size, contamination)
            test = prepare_reuters(test, split_size, contamination)
        elif dataset == 'newsgroups':
            train = prepare_newsgroups(train, split_size, contamination)
            test = prepare_newsgroups(test, split_size, contamination)

        lsa_tr_tmp, lsa_te_tmp = run_models(
            train, test, contamination=contamination
        )

        lsa_tr[rank] = lsa_tr_tmp
        lsa_te[rank] = lsa_te_tmp

    logger.info("Experiment on LSA -Done-")

    # ---------------------------------------------------------
    # -======================== VADER ========================-
    # ---------------------------------------------------------
    logger.info("=====VADER=====")

    train = data['train_vader'].copy()[['neg', 'pos']]
    train['label'] = data['train']['label']
    train['clean'] = data['train']['clean']

    test = data['test_vader'].copy()[['neg', 'pos']]
    test['label'] = data['test']['label']
    test['clean'] = data['test']['clean']

    if dataset == 'reuters':
        train = prepare_reuters(train, split_size, contamination)
        test = prepare_reuters(test, split_size, contamination)
    elif dataset == 'newsgroups':
        train = prepare_newsgroups(train, split_size, contamination)
        test = prepare_newsgroups(test, split_size, contamination)

    vader_tr, vader_te = run_models(train, test, contamination=contamination)

    logger.info("Experiment on VADER -Done-")

    # ---------------------------------------------------------
    # -========================= OUR =========================-
    # ---------------------------------------------------------
    logger.info("=====OUR=====")

    our_tr = {}
    our_te = {}

    for rank in LSA_K:
        logger.info("=====LSA-SE-{0}=====".format(rank))

        train = pd.DataFrame(data['train_lsa_{}'.format(rank)])
        train['clean'] = data['train']['clean']
        test = pd.DataFrame(data['test_lsa_{}'.format(rank)])
        test['clean'] = data['test']['clean']

        for pol in ['neg', 'pos']:
            train[pol] = data['train_vader'][pol]
            test[pol] = data['test_vader'][pol]

        train['label'] = data['train']['label']
        test['label'] = data['test']['label']

        if dataset == 'reuters':
            train = prepare_reuters(train, split_size, contamination)
            test = prepare_reuters(test, split_size, contamination)
        elif dataset == 'newsgroups':
            train = prepare_newsgroups(train, split_size, contamination)
            test = prepare_newsgroups(test, split_size, contamination)

        our_tr_tmp, our_te_tmp = run_models(
            train, test, contamination=contamination
        )

        our_tr[rank] = our_tr_tmp
        our_te[rank] = our_te_tmp

    logger.info("Experiment on OUR approach -Done-")

    return {
        'tfidf_train': tfidf_tr,
        'tfidf_test': tfidf_te,
        'lsa_train': lsa_tr,
        'lsa_test': lsa_te,
        'vader_train': vader_tr,
        'vader_test': vader_te,
        'our_train': our_tr,
        'our_test': our_te
    }


def run_models(train,
            test,
            metric="manhattan",
            kernel="rbf",
            degree=2,
            contamination=.1):
    logger = logging.getLogger(__name__)

    lof = LOF(n_jobs=-1, metric=metric, contamination=contamination)
    ocsvm = OCSVM(kernel=kernel, degree=degree, contamination=contamination)
    iforest = IForest(n_jobs=-1, contamination=contamination)

    res_tr = {
        "class": [], "auroc_lof": [], "ap_lof": [],
        "auroc_ocsvm": [], "ap_ocsvm": [], "auroc_ifo": [], 'ap_ifo': []
    }
    res_te = {
        "class": [], "auroc_lof": [], "ap_lof": [],
        "auroc_ocsvm": [], "ap_ocsvm": [], "auroc_ifo": [], 'ap_ifo': []
    }

    for k in train.keys():
        res_tr['class'].append(k)
        res_te['class'].append(k)

        # train split
        x = train[k].drop(['label'], axis=1)
        y = [1 if ls != k else 0 for ls in train[k]['label']]

        # test split
        xx = test[k].drop(['label'], axis=1)
        yy = [1 if ls != k else 0 for ls in test[k]['label']]

        logger.info("CLASS == {}".format(k))

        lof.fit(x)
        evaluate_model(x, lof, 'lof', y, res_tr)
        evaluate_model(xx, lof, 'lof', yy, res_te)

        # logger.info("LOF -Done-")

        ocsvm.fit(x)
        evaluate_model(x, ocsvm, 'ocsvm', y, res_tr)
        evaluate_model(xx, ocsvm, 'ocsvm', yy, res_te)

        # logger.info("OCSVM -Done-")

        iforest.fit(x)
        evaluate_model(x, iforest, 'ifo', y, res_tr)
        evaluate_model(xx, iforest, 'ifo', yy, res_te)

        # logger.info("IForest -Done-")

    return pd.DataFrame(res_tr), pd.DataFrame(res_te)


def evaluate_model(data, model, model_name, y, res):
    f, t, th = roc_curve(y, [o for i, o in model.predict_proba(data)])

    auroc = auc(f, t)
    ap = average_precision_score(y, [o for i, o in model.predict_proba(data)])

    res['auroc_{}'.format(model_name)].append(auroc)
    res['ap_{}'.format(model_name)].append(ap)
