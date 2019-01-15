import twitter


def build_corpus(username, api):
    """
    Build the corpus using a twitter name
    :return: (String) corpus of training data
    """
    print('getting tweets for user: ', username)
    timeline = api.GetUserTimeline(screen_name=username, count=200)
    tweets = [t.text for t in timeline]
    corpus = ' '.join(tweets)
    return corpus


def get_keys():
    """
    Get the Keys for Twitter account authentication
    :return: (Tuple) consumer_api_key, consumer_secret_key, access_token, access_secret_token
    """
    keys = []
    with open('keys', 'r') as file:
        for line in file:
            keys.append(line.strip('\n'))
    return tuple(keys)


def get_api():
    """
    Get the twitter API
    :return: Twitter API object
    """
    consumer_api_key, consumer_secret_key, access_token, access_secret_token = get_keys()
    return twitter.Api(consumer_key=consumer_api_key,
                       consumer_secret=consumer_secret_key,
                       access_token_key=access_token,
                       access_token_secret=access_secret_token)
