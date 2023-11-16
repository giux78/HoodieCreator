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

def create_product(user, body) -> str:
    b64 = body['imagebase64']
    im = Image.open(BytesIO(base64.b64decode(b64)))
    imgbk = Image.open(r"./data/hoodie-black-retro.png") 

    img3 = im.resize((300,300))
  
    imgbk.paste(img3, (250,280)) 
    imgbk.save(r"./static/test.png")
    ENV_URL = os.getenv('ENV_URL')
    image_url = f'{ENV_URL}/static/test.png'
    image_front = f'{ENV_URL}/static/hoodie-black-front.png'
    print(image_url)
    description_stripe = '80x80 cm black wood framed print send directly to your home'
    
    metadata_stripe = {"size" : "M", 
                        "color" :  'black', 
                        "path_image" : './data/test.png',
                        }
    
    try:
        product = stripe.Product.create(name='test', 
                                description=description_stripe, 
                                images=[image_url,image_front],   
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
    
'''
    try:
        product = stripe.Product.create(name=name, 
                                description=description_stripe, 
                                images=[url_image],   
                                default_price_data={"unit_amount": 19900, "currency": "eur"},
                                metadata=metadata_stripe)
        print(product)
        product_id = product['id']
        price= stripe.Price.create(
            unit_amount=19900,
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
            success_url=YOUR_DOMAIN + '/success.html',
            cancel_url=YOUR_DOMAIN + '/cancel.html',
                shipping_address_collection={
                    'allowed_countries': ['IT'],
                }
        )
      
        metadata = NFTMetadataInput.from_json({
                    "name": name_nft,
                    "description": description_nft,
                    "image": image_nft
                })
                
                # 0xB3053C6baAc7eAC6e0F9037f1E2C92a0f5f6e094
        tx = current_app.nft_module.mint_to("0xB3053C6baAc7eAC6e0F9037f1E2C92a0f5f6e094", metadata)
        receipt = tx.receipt
        token_id = tx.id
        nft = tx.data()
        print(nft)

        metadata_stripe["id_nft"] = str(token_id)
        stripe.Product.modify(
                product_id,
                metadata=metadata_stripe,
            )
  except Exception as e:
        print(str(e))
        return str(e)
    return  {'link_stripe' : checkout_session.url}
'''


app = connexion.FlaskApp(__name__, specification_dir="spec", )
app.add_api("openapi.json")
load_dotenv()

stripe.api_key  = os.getenv('STRIPE_SECRET_KEY')


if __name__ == "__main__":
    app.run(f"{Path(__file__).stem}:app", port=80)
