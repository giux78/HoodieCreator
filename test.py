
from PIL import Image 
import os
from dotenv import load_dotenv
import urllib.request
import requests 
import re

'''
urllib.request.urlretrieve( 
  'https://oaidalleapiprodscus.blob.core.windows.net/private/org-LVFFcnUSRUHlXMzkIYSvX3eq/user-GY3gwrD1ESuSDZnOL0gtktJH/img-GEndZ2n8AclYchC7UbpwPv7N.png?st=2023-11-17T18%3A47%3A45Z&se=2023-11-17T20%3A47%3A45Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-11-17T02%3A30%3A14Z&ske=2023-11-18T02%3A30%3A14Z&sks=b&skv=2021-08-06&sig=04CFtU%2BZ%2BIkMofcKuTerCw%2BLbGUKG3Se1aKhhuDiDeM%3D', 
   "test.png") 
  
img = Image.open("test.png") 
img.show()
'''

load_dotenv()

from openai import OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_KEY"))
print(client.files.list())
#client.files.retrieve("file-iWtdUVELk3tPISWLcBkRuh6Z")

from PIL import Image
import base64
from io import BytesIO

# Load the image
'''
image_path = './data/lynx.png'
image = Image.open(image_path)


buffered = BytesIO()
image.save(buffered, format="PNG")
img_str = base64.b64encode(buffered.getvalue()).decode()

print(img_str)  # Displaying the first 100 characters of the base64 string as a sanity check
'''

''' 
img1 = Image.open(r"./data/hoodie-black-front.png") 
img2 = Image.open(r"./data/qrcode_test.png") 

img3 = img2.resize((50,50))
  
# No transparency mask specified,  
# simulating an raster overlay 
img1.paste(img3, (375,250)) 
  
img1.show()
'''
'''
You are an artist and must help to create a wonderful image.  Please do not create image of hoodie
Ask the user if he loves the image and in case he say yes ask for color, size, and name. 
Once you know all the information call di action passing a json with name, color, and imagebase64 keys.
You must always transform every image into base64 format without truncating the base64 string no [:100] please
''' 

#API_KEY=os6P0xhzhFGXRQIXSlWFfQphq
#API_SECRET=AZMu9n00sSd5ZuYzvNV7YZO17Fewg4mJb9uR7PJ5vg1zxKgXEJ
#BEARER_TOKEN=AAAAAAAAAAAAAAAAAAAAAMHsrAEAAAAAWpfaN4eHwdv3u1scPLBTbmEbTiE%3Dt5eqTYj2HJt349f4EWLF1BOclnlGhQAu4jfsgz91A0QjqzenC9
#ACCESS_TOKEN=1729046426141736960-6diD9K5LzRvj5ZyZxgNOR3f4wK7ntJ
#ACCESS_TOKEN_SECRET=t5ZAsh9IG4prHOYLF5i4dnRN9ReoofcbbLG4FRdVxKACf

import tweepy


consumer_key = "os6P0xhzhFGXRQIXSlWFfQphq"
consumer_secret = "AZMu9n00sSd5ZuYzvNV7YZO17Fewg4mJb9uR7PJ5vg1zxKgXEJ"
access_token = "1729046426141736960-VcDj6knvjMzxfuge0Pw56sF0UIFncA"
access_token_secret = "H92GN5Rvk6p34WiyKuC0ToIUL1l6ImWZoL4tOzof1EhzT"

CLIENT_ID = "YUJZRUl2cFpFLXN2aW92a0lSRXM6MTpjaQ"
CLIENT_SECRET= "6zvD0BOIDiR3vApjlnIwN8z-vHDhT3ruCqgWu1TeER3uE5rB0a"

client = tweepy.Client(
    consumer_key=consumer_key, consumer_secret=consumer_secret,
    access_token=access_token, access_token_secret=access_token_secret
)

# Create Tweet

# The app and the corresponding credentials must have the Write permission

# Check the App permissions section of the Settings tab of your app, under the
# Twitter Developer Portal Projects & Apps page at
# https://developer.twitter.com/en/portal/projects-and-apps

# Make sure to reauthorize your app / regenerate your access token and secret 
# after setting the Write permission

#response = client.create_tweet(
#    text="Here my first product created",
#)
#print(f"https://twitter.com/user/status/{response.data['id']}")

def upload_media(image_url):
    tweepy_auth = tweepy.OAuth1UserHandler(
        "{}".format(consumer_key),
        "{}".format(consumer_secret),
        "{}".format(access_token),
        "{}".format(access_token_secret),
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

media_obj = upload_media("https://hoodie-creator.s3.eu-west-1.amazonaws.com/d41f8cb3.png")

response = client.create_tweet(
    text="Here my first automatic product",
    media_ids= media_obj['media']['media_ids']
)

print(f"https://twitter.com/user/status/{response.data['id']}")