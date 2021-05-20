from src import data_loader

REUTERS_LARGEST = [
    data_loader.REUTERS_CATEGORIES['earn'],
    data_loader.REUTERS_CATEGORIES['acq'],
    data_loader.REUTERS_CATEGORIES['money-fx'],
    data_loader.REUTERS_CATEGORIES['crude'],
    data_loader.REUTERS_CATEGORIES['trade']
]


def prepare_reuters(reuters, split_size=-1, contamination=0.1):
    df = reuters.copy()

    df = df[df['label'].str.len() == 1]

    df['label'] = df['label'].apply(lambda l: l[0])

    splits = {}

    for c in REUTERS_LARGEST:
        splits[c] = sample_contamination(df, split_size, c, contamination)

    return splits


def prepare_newsgroups(newsgroups, split_size=-1, contamination=0.1):
    df = newsgroups.copy()

    # we avoid the case where body of the message is empty
    df = df[df['clean'].astype(bool)]

    splits = {}

    for c in df['label'].unique():
        splits[c] = sample_contamination(df, split_size, c, contamination)

    return splits


def sample_contamination(data, split_size, clss, contamination):
    if split_size == -1:
        tmp = data[data['label'] == clss]

        total_outliers = int(contamination * len(tmp))
        if total_outliers == 0:
            total_outliers = 1

        tmp = tmp.sample(len(tmp) - total_outliers)
    else:
        total_outliers = int(contamination * split_size)
        if total_outliers == 0:
            total_outliers = 1

        tmp = data[data['label'] == clss].sample(split_size - total_outliers)

    tmp = tmp.append(data[data['label'] != clss].sample(total_outliers))

    return tmp.sample(frac=1)
