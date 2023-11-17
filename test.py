
from PIL import Image 
import os
from dotenv import load_dotenv


load_dotenv()

from openai import OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_KEY"))
print(client.files.list())
#client.files.retrieve("file-iWtdUVELk3tPISWLcBkRuh6Z")

from PIL import Image
import base64
from io import BytesIO

# Load the image
image_path = './data/lynx.png'
image = Image.open(image_path)


buffered = BytesIO()
image.save(buffered, format="PNG")
img_str = base64.b64encode(buffered.getvalue()).decode()

print(img_str)  # Displaying the first 100 characters of the base64 string as a sanity check




'''  
img1 = Image.open(r"./data/hoodie-black-retro.png") 
img2 = Image.open(r"./data/lynx.png") 

img3 = img2.resize((300,300))
  
# No transparency mask specified,  
# simulating an raster overlay 
img1.paste(img3, (250,280)) 
  
img1.show()
''' 