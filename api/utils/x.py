import requests 
import re
import tweepy
import os

def upload_media(image_url):
    tweepy_auth = tweepy.OAuth1UserHandler(
        "{}".format(os.getenv('X_CONSUMER_KEY')),
        "{}".format(os.getenv('X_CONSUMER_SECRET')),
        "{}".format(os.getenv('X_ACCESS_TOKEN')),
        "{}".format(os.getenv('X_ACCESS_TOKEN_SECRET')),
    )

    name = image_url.rsplit('/', 1)[-1]
    tweepy_api = tweepy.API(tweepy_auth)
    img_data = requests.get(image_url).content
    with open(f"/tmp/{name}", "wb") as handler:
        handler.write(img_data)
    post = tweepy_api.simple_upload(f"/tmp/{name}")
    text = str(post)
    media_id = re.search("media_id=(.+?),", text).group(1)
    payload = {"media": {"media_ids": ["{}".format(media_id)]}}
    os.remove(f"/tmp/{name}")
    return payload