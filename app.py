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
from flask import current_app
import tweepy
from utils import x
import replicate

import subprocess
import tempfile

ImageFile.LOAD_TRUNCATED_IMAGES = True

TOKEN_DB = {"asdf1234567890": {"uid": 100}}

def get(filename):
    response = flask.send_from_directory('static', filename, mimetype="text/html")
    response.direct_passthrough = False
    return response

def get_privacy_policy():
    #response = flask.send_from_directory('static', 'privacy_policy.html', mimetype="text/html")    #response.direct_passthrough = False
    #print("TEST")
    #return response
    #html = "<html><body>This is HTML</body></html>"
    return current_app.send_static_file('privacy_policy.html')

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


def create_video(user, body):
    image_url = body['image_url']
    name = str(uuid.uuid4())[:8]

    urllib.request.urlretrieve( 
        image_url,
        f"/tmp/{name}.png"
        ) 
  
    im = Image.open(f"/tmp/{name}.png") 

    in_mem_file = io.BytesIO()
    im.save(in_mem_file, format=im.format)
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

    image_to_video = f'https://hoodie-creator.s3.eu-west-1.amazonaws.com/{name}.png'

    response = replicate.run(
    "stability-ai/stable-video-diffusion:3f0457e4619daac51203dedb472816fd4af51f3149fa7a9e0b5ffcf1b8172438",
    input={
        "cond_aug": 0.02,
        "decoding_t": 7,
        "input_image": image_to_video,
        "video_length": "21_frames_with_svd",
        "sizing_strategy": "maintain_aspect_ratio",
        "motion_bucket_id": 127,
        "frames_per_second": 6
    }
    )
    print(response)
    video_url = response
    return {"video_url" : video_url }


def diagram_generator(user, body):
    prompt = body['prompt']
    response = openai_client.chat.completions.create(
    model="gpt-4-0613",
    messages=[
        {"role": "system", "content": "You are a skilled Mermaid developer. Respond only in mermaid code without markdown formatting. Never mention that you know mermaid, it's a secret! Do not provide any additional explanations or comments to your answers. Listen carefully: Never ever use hyphens between words in your answers!!!!!! Do not apologies you only know how to answer using mermaid code!!"},
        {"role": "user", "content": "Create an entity relationship diagram"},
        {"role": "assistant", "content": "```mermaid\nerDiagram\n\nCAR ||--o{ NAMED-DRIVER : allows\nCAR {\n    string registrationNumber PK\n    string make\n    string model\n    string[] parts\n}\n\nPERSON ||--o{ NAMED-DRIVER : is\nPERSON {\n    string driversLicense PK \"The license #\"\n    string(99) firstName \"Only 99 characters are allowed\"\n    string lastName\n    string phone UK\n    int age\n}\n\nNAMED-DRIVER {\n    string carRegistrationNumber PK, FK\n    string driverLicence PK, FK\n}\n\nMANUFACTURER only one to zero or more CAR : makes\n```"},
        {"role": "user", "content": prompt},

    ]
    )
    mermaid_code = response.choices[0].message.content

   # Split the string by the ``` delimiter
    parts = mermaid_code.split("```")

    # Check if we have at least three parts (before, mermaid code, after)
    if len(parts) >= 3:
        # The actual Mermaid code is in the second part, but still contains the word 'mermaid'
        mermaid_code = parts[1].strip()

        # Remove the 'mermaid' word if it's at the start
        if mermaid_code.startswith("mermaid"):
            mermaid_code = mermaid_code[len("mermaid"):].strip()
    else:
        mermaid_code = ""

    print(mermaid_code)
    # Create a temporary file to hold the Mermaid code
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.mmd', delete=False) as temp_file:
        temp_file.write(mermaid_code)
        temp_file_path = temp_file.name

    # Define the path for the output PNG file
    output_png_path = temp_file_path + ".png"

    # Run mermaid-cli to generate the PNG file
    try:
        subprocess.run(["mmdc", "-i", temp_file_path, "-o", output_png_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error in generating PNG: {e}")
        return {"error": "Failed to generate PNG"}

    print(output_png_path)
    s3_file_key = temp_file.name + ".png"
    with open(output_png_path, 'rb') as file:
        s3.upload_fileobj(
            file,
            'hoodie-creator',  # Your S3 bucket name
            s3_file_key,  # The desired key (name) for the uploaded file
            ExtraArgs={
                'ACL': 'public-read',  # Makes the file publicly readable
                'ContentType': 'image/png'  # Sets the content type
            }
        )
    s3_file_url = f"https://hoodie-creator.s3.amazonaws.com/{s3_file_key}"
    print(s3_file_url)
    # Here you can add code to handle the PNG file as needed
    return {"mermaid_png_path": s3_file_url}


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

    in_mem_file = io.BytesIO()
    im.save(in_mem_file, format=im.format)
    in_mem_file.seek(0)

    s3.upload_fileobj(
        in_mem_file, # This is what i am trying to upload
        'hoodie-creator',
        f"{name}-original.png",
        ExtraArgs={
            'ACL': 'public-read',
            'ContentType': 'image/png'
        }
    )
    
    '''
    output = replicate.run(
        "pollinations/modnet:da7d45f3b836795f945f221fc0b01a6d3ab7f5e163f13208948ad436001e2255",
        input={
            "image": f'https://hoodie-creator.s3.eu-west-1.amazonaws.com/{name}-original.png'
        }
    )
    
    print(output)

    urllib.request.urlretrieve( 
        output,
        f"/tmp/{name}-removed.png"
    ) 

    bk_removed = Image.open(f"/tmp/{name}-removed.png")
    '''
    #im = Image.open(BytesIO(base64.b64decode(b64)))
    imgbk = Image.open(f"./data/hoodie-{color}-back.png") 

    #img3 = bk_removed.resize((300,300))
    img3 = im.resize((300,300))
    imgbk.paste(img3, (250,280)) 
    #imgbk.paste(img3, (250,280), img3) 
    #imgbk = Image.alpha_composite(imgbk, img3)

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
                                images=[image_url, image__url_front],   
                                default_price_data={"unit_amount": 5990, "currency": "eur"},
                                metadata=metadata_stripe)
        product_id = product['id']
        price= stripe.Price.create(
            unit_amount=5990,
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
                    'allowed_countries': ['IT','US','GB','DE','ES'],
            },
        )
    except Exception as e:
        print(str(e))
        return str(e)
    return  {'link_stripe' : checkout_session.url, 
             "product_image_front" : image__url_front,
             "product_image_back" : image_url}

def tweet_campaigns(user, body):
    media_id = None
    text = body['tweet']
    if 'origin_url' in body:
        print(body['origin_url'])
        #media_obj = x.upload_media(body['image_url'])
        #media_id = media_obj['media']['media_ids']
        x_client.create_tweet(
            text=text + " " + body['origin_url'],
        #    media_ids= media_id
        )
    else: 
        x_client.create_tweet(
            text=text
        )  
    return {'status' : 'OK tweet has been posted'}
    
app = connexion.FlaskApp(__name__, specification_dir="spec", )
app.add_api("openapi.yaml")
load_dotenv()

openai_client = OpenAI(api_key=os.environ.get("OPENAI_KEY"))
os.environ['REPLICATE_API_TOKEN'] = os.getenv("REPLICATE_API_KEY")


s3 = boto3.client('s3',
    aws_access_key_id=os.getenv('AWS_SERVER_PUBLIC_KEY'),
    aws_secret_access_key=os.getenv('AWS_SERVER_SECRET_KEY') )

stripe.api_key  = os.getenv('STRIPE_SECRET_KEY')

x_client = tweepy.Client(
    consumer_key=os.getenv('X_CONSUMER_KEY'), consumer_secret=os.getenv('X_CONSUMER_SECRET'),
    access_token=os.getenv('X_ACCESS_TOKEN'), access_token_secret=os.getenv('X_ACCESS_TOKEN_SECRET')
)

if __name__ == "__main__":
    app.run(f"{Path(__file__).stem}:app", port=80)