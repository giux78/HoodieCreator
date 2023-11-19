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
import uuid

 


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
    size = body['size']
    prompt = body['prompt']

    name = str(uuid.uuid4())[:8]

    urllib.request.urlretrieve( 
        image_url,
        f"/tmp/{name}.png"
        ) 
  
    im = Image.open(f"/tmp/{name}.png") 

    #im = Image.open(BytesIO(base64.b64decode(b64)))
    imgbk = Image.open(f"./data/hoodie-{color}-back.png") 

    img3 = im.resize((300,300))
    imgbk.paste(img3, (250,280)) 

    img_front = Image.open(f"./data/hoodie-{color}-front.png")
    img_qr = Image.open(r"./data/qrcode_test.png")
    img_qr = img_qr.resize((50,50))
    img_front.paste(img_qr, (375,250)) 

    ENV_URL = os.getenv('ENV_URL')
    image_url = f'https://hoodie-creator.s3.eu-west-1.amazonaws.com/{name}.png'
    #image_front = f'{ENV_URL}/static/hoodie-black-front.png'
    image__url_front = f'https://hoodie-creator.s3.eu-west-1.amazonaws.com/{name}-front.png'
    print(image_url)
    description_stripe = 'hoodie'
    
    metadata_stripe = {"size" : size, 
                        "color" :  color, 
                        "url_image" : image_url,
                        "prompt" : prompt
                        }
    
    in_mem_file = io.BytesIO()
    imgbk.save(in_mem_file, format=imgbk.format)
    in_mem_file.seek(0)

    s3.upload_fileobj(
        in_mem_file, # This is what i am trying to upload
        'hoodie-creator',
        f"{name}.png",
        ExtraArgs={
            'ACL': 'public-read',
            'ContentType': 'image/png'
        }
    )

    in_mem_file = io.BytesIO()
    img_front.save(in_mem_file, format=imgbk.format)
    in_mem_file.seek(0)

    s3.upload_fileobj(
        in_mem_file, # This is what i am trying to upload
        'hoodie-creator',
        f"{name}-front.png",
        ExtraArgs={
            'ACL': 'public-read',
            'ContentType': 'image/png'
        }
    )
    
    try:
        product = stripe.Product.create(name=name, 
                                description=description_stripe, 
                                images=[image_url, image_front],   
                                default_price_data={"unit_amount": 8990, "currency": "eur"},
                                metadata=metadata_stripe)
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
    return  {'link_stripe' : checkout_session.url, 
             "product_image_front" : image__url_front,
             "product_image_back" : image_url}
    
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
