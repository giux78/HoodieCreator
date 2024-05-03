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
import requests
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
from transformers import AutoTokenizer
import json

from upstash_redis import Redis
from upstash_ratelimit import FixedWindow, Ratelimit

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
    #info = TOKEN_DB.get(token, None)
    user = None
    if(redis.exists(f'api:{token}')):
        user = redis.hgetall(f'api:{token}')
        print(user)
    elif token == "asdf1234567890":
        user = { "n_token" : 10000, "uid" : "apikey" }
    
    if not user:
        raise OAuthProblem("Invalid token")
    print(user)
    return user


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

    name = str(uuid.uuid4())[:8]

    urllib.request.urlretrieve( 
        image_url,
        f"/tmp/{name}-public.png"
        ) 
  
    im = Image.open(f"/tmp/{name}-public.png") 

    in_mem_file = io.BytesIO()
    im.save(in_mem_file, format=im.format)
    in_mem_file.seek(0)

    s3.upload_fileobj(
        in_mem_file, # This is what i am trying to upload
        'hoodie-creator',
        f"{name}-public.png",
        ExtraArgs={
            'ACL': 'public-read',
            'ContentType': 'image/png'
        }
    )

    image_s3 = f'https://hoodie-creator.s3.eu-west-1.amazonaws.com/{name}-public.png'

    return {"image_url" : image_s3 }

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
        "video_length": "14_frames_with_svd",
        "sizing_strategy": "maintain_aspect_ratio",
        "motion_bucket_id": 127,
        "frames_per_second": 6
    }
    )
    print(response)
    video_url = response
    return {"video_url" : video_url }

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
             "product_image_back" : image_url,
             "price_id" : price_id}

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

def zefiro_generate(user, body):
    messages = body
    print(messages)
    sys_prompt = "Sei un assistente disponibile, rispettoso e onesto. " 
          
        # "Rispondi sempre nel modo piu' utile possibile, pur essendo sicuro. " \
        # "Le risposte non devono includere contenuti dannosi, non etici, razzisti, sessisti, tossici, pericolosi o illegali. " \
        # "Assicurati che le tue risposte siano socialmente imparziali e positive. " \
        # "Se una domanda non ha senso o non e' coerente con i fatti, spiegane il motivo invece di rispondere in modo non corretto. " \
        # "Se non conosci la risposta a una domanda, non condividere informazioni false."
    apikey = redis.get(f'user:{user}:api')
    limiting = ratelimit.limit(apikey)
    if not limiting.allowed:
        return {'error' : 'too many request per minute'}, 429


    if(messages[0]['id'] != 'sys'):
        messages.insert(0, {'role' : 'assistant', 'content' : sys_prompt, "id" : "sys"})
    
    '''
    to_generate = tokenizer.apply_chat_template(messages , 
                                                tokenize=False, 
                                                add_generation_prompt=True
                                                #truncation=False, 
                                                #padding_side='left'
                                                )
    '''
    # Zefiro 0.7
    # API_URL = "https://uqa65rd8kujtn7lw.us-east-1.aws.endpoints.huggingface.cloud"
    #zefiro 0.5
    #API_URL = "https://h2opl5lmg1oqd2o2.us-east-1.aws.endpoints.huggingface.cloud"
    #zefiro 0.1
    #API_URL="https://lbqf6xe9jk6h0z2q.us-east-1.aws.endpoints.huggingface.cloud"
    #API_URL="http://ec204616.seewebcloud.it:30015/generate"
    API_URL="https://zefiro-api.seeweb.ai/v1/chat/completions"
    '''
    curl -i -H --location $ENDPOINT \
	     --header "Content-Type: application/json" \
	     --header "Accept: application/json" \
	     --header "Authorization: Bearer ${APIKEY}" \
	     --data   '{
	               "model": "giux78/zefiro-7b-dpo-qlora-ITA-v0.7",
                   "messages": [ {
		          "role": "user",
			  "content": "Parlami di Roma, della sua storia, delle sue bellezze"
			}]
		       }'
        '''


    headers = {
        "Accept" : "application/json",
        "Authorization": "Bearer " + os.environ.get('SEEWEB'),
        "Content-Type": "application/json" 
    }

    payload = {
	               "model": "giux78/zefiro-7b-dpo-qlora-ITA-v0.7",
                   "messages": messages,
                   "max_new_tokens": 1024, 
                    "skip_special_tokens": True,
                    "no_repeat_ngram_size": 5,
                    "repetition_penalty" : 1.2
		       }
    ''' 
    payload = {
	    "inputs": to_generate,
	    "parameters": {"max_new_tokens": 1024, 
                       "skip_special_tokens": True,
                       "no_repeat_ngram_size": 5,
                       "repetition_penalty" : 1.2
                       }
    }
    '''
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        generated_resp = response.json()
        print(generated_resp)
        # worked on hf inference endpoint
        #generated_text = generated_resp[0]['generated_text'].replace('<|assistant|>','')
        # work on seeweb
        #generated_resp['generated_text'].replace('<|assistant|>','')
        generated_text = generated_resp['choices'][0]['message']['content'] 
        print(generated_text)
        messages.append({'role' : 'assistant', 'content' : generated_text})
    else:
        messages.append({'role' : 'assistant', 'content' : "aspetta circa un minuto che stiamo attivando le gpus necessarie"})
    
    if isinstance(messages, list):
        tokenFromMess = 0
        for message in messages:
            tokenFromMess += round(len(message['content'].split(' ')) * 2 * 1.5)
        
        # Assuming kv is an instance of an async key-value store library
        info = redis.hgetall(f"api:{apikey}")
        nToken = 100000
        if "n_token" in info:
            nToken = int(info['n_token'])
        
        count = nToken - tokenFromMess
        redis.hset(f"api:{apikey}", 'n_token', count)
        
        # Remove system prompt
        messages.pop(0)

    return messages


    
app = connexion.FlaskApp(__name__, specification_dir="spec", )
app.add_api("openapi.yaml")
load_dotenv()

openai_client = OpenAI(api_key=os.environ.get("OPENAI_KEY"))
os.environ['REPLICATE_API_TOKEN'] = os.getenv("REPLICATE_API_KEY")

redis = Redis(url=os.getenv('UPSTASH_REDIS_REST_URL'), token=os.getenv("UPSTASH_REDIS_REST_TOKEN"))

ratelimit = Ratelimit(
    redis=redis,
    limiter=FixedWindow(max_requests=3, window=60),
)


s3 = boto3.client('s3',
    aws_access_key_id=os.getenv('AWS_SERVER_PUBLIC_KEY'),
    aws_secret_access_key=os.getenv('AWS_SERVER_SECRET_KEY') )

tokenizer = AutoTokenizer.from_pretrained("giux78/zefiro-7b-sft-qlora-ITA-v0.5", 
                                          padding_side="left")

stripe.api_key  = os.getenv('STRIPE_SECRET_KEY')

x_client = tweepy.Client(
    consumer_key=os.getenv('X_CONSUMER_KEY'), consumer_secret=os.getenv('X_CONSUMER_SECRET'),
    access_token=os.getenv('X_ACCESS_TOKEN'), access_token_secret=os.getenv('X_ACCESS_TOKEN_SECRET')
)

if __name__ == "__main__":
    app.run(f"{Path(__file__).stem}:app", port=80)