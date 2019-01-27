from util import *
from model import ModelGen
import argparse
from lstm_gen import train_model


def main():
    parser = argparse.ArgumentParser(description='Train a generative model on tweets')
    parser.add_argument('-u', type=str, help='Twitter username')
    args = parser.parse_args()
    username = args.u

    api = get_api()

    corpus = build_corpus(username, api)
    train_model(corpus)


if __name__ == '__main__':
    main()
