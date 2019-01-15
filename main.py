from util import *
from model import ModelGen


def main():
    api = get_api()
    corpus = build_corpus('realDonaldTrump', api)
    ModelGen().train_model(corpus)


if __name__ == '__main__':
    main()
