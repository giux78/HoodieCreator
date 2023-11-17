"""
Basic example of a resource server
"""
from pathlib import Path
import os

import connexion
from connexion.exceptions import OAuthProblem

from PIL import Image
from io import BytesIO
import base64
import os
from dotenv import load_dotenv
import stripe
import flask
import boto3
import io
from openai import OpenAI
from PIL import ImageFile
import urllib.request


ImageFile.LOAD_TRUNCATED_IMAGES = True

TOKEN_DB = {"asdf1234567890": {"uid": 100}}

def get(filename):
    response = flask.send_from_directory('static', filename)
    response.direct_passthrough = False
    return response

def apikey_auth(token, required_scopes):
    info = TOKEN_DB.get(token, None)

    if not info:
        raise OAuthProblem("Invalid token")

    return info


def get_secret(user) -> str:
    return f"You are {user} and the secret is 'wbevuec'"

def create_image(user, body):
    prompt = body['prompt']

    response = openai_client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )

    image_url = response.data[0].url
    return {"image_url" : image_url }

def create_product(user, body) -> str:
    image_url = body['image_url']
    color = body['color']
    name = body['name']

    urllib.request.urlretrieve( 
        "image_url"
        "test.png"
        ) 
  
    im = Image.open("test.png") 

    #im = Image.open(BytesIO(base64.b64decode(b64)))
    imgbk = Image.open(r"./data/hoodie-black-retro.png") 

    img3 = im.resize((300,300))
    imgbk.paste(img3, (250,280)) 

    ENV_URL = os.getenv('ENV_URL')
    image_url = f'https://hoodie-creator.s3.eu-west-1.amazonaws.com/{name}'
    #image_front = f'{ENV_URL}/static/hoodie-black-front.png'
    print(image_url)
    description_stripe = 'hoodie'
    
    metadata_stripe = {"size" : "M", 
                        "color" :  'black', 
                        "url_image" : image_url,
                        }
    
    in_mem_file = io.BytesIO()
    imgbk.save(in_mem_file, format=imgbk.format)
    in_mem_file.seek(0)

    s3.upload_fileobj(
        in_mem_file, # This is what i am trying to upload
        'hoodie-creator',
        name,
        ExtraArgs={
            'ACL': 'public-read',
            'ContentType': 'image/png'
        }
    )
    
    try:
        product = stripe.Product.create(name=name, 
                                description=description_stripe, 
                                images=[image_url, 'https://hoodie-creator.s3.eu-west-1.amazonaws.com/hoodie-black-front.png'],   
                                default_price_data={"unit_amount": 8990, "currency": "eur"},
                                metadata=metadata_stripe)
        print(product)
        product_id = product['id']
        price= stripe.Price.create(
            unit_amount=8990,
            currency="eur",
            product=product_id,
        )
        print(price)
        price_id = price['id']
        checkout_session = stripe.checkout.Session.create(
            line_items=[
                {
                    # Provide the exact Price ID (for example, pr_1234) of the product you want to sell
                    'price': price_id,
                    'quantity': 1,
                },
            ],
            mode='payment',
            success_url= f'{ENV_URL}/static/success.html',
            cancel_url= f'{ENV_URL}/static/cancel.html',
                shipping_address_collection={
                    'allowed_countries': ['IT','US','GB'],
                }
        )
    except Exception as e:
        print(str(e))
        return str(e)
    return  {'link_stripe' : checkout_session.url}
    
app = connexion.FlaskApp(__name__, specification_dir="spec", )
app.add_api("openapi.yaml")
load_dotenv()

openai_client = OpenAI(api_key=os.environ.get("OPENAI_KEY"))

s3 = boto3.client('s3',
    aws_access_key_id=os.getenv('AWS_SERVER_PUBLIC_KEY'),
    aws_secret_access_key=os.getenv('AWS_SERVER_SECRET_KEY') )

stripe.api_key  = os.getenv('STRIPE_SECRET_KEY')

if __name__ == "__main__":
    app.run(f"{Path(__file__).stem}:app", port=80)
