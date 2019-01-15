import twitter
from model import ModelGen


def main():
    api = get_Api()
    corpus = build_corpus('realDonaldTrump', api)

    # t = api.GetUserTimeline(screen_name="realDonaldTrump", count=500)


def build_corpus(username, api):
    """
    Build the corpus using a twitter name
    :return: (String) corpus of training data
    """
    print('getting tweets for user: ', username)
    # user = api.GetUser(screen_name=username)
    timeline = api.GetUserTimeline(screen_name=username, count=200)
    print(timeline[0])

    return ''

def get_Keys():
    """
    Get the Keys for Twitter account authentication
    :return: (Tuple) consumer_api_key, consumer_secret_key, access_token, access_secret_token
    """
    keys = []
    with open('keys', 'r') as file:
        for line in file:
            keys.append(line.strip('\n'))
    return tuple(keys)


def get_Api():
    """
    Get the twitter API
    :return: Twitter API object
    """
    consumer_api_key, consumer_secret_key, access_token, access_secret_token = get_Keys()
    return twitter.Api(consumer_key=consumer_api_key,
                       consumer_secret=consumer_secret_key,
                       access_token_key=access_token,
                       access_token_secret=access_secret_token)


if __name__ == '__main__':
    main()
