
from PIL import Image 
import os
from dotenv import load_dotenv
import urllib.request 

urllib.request.urlretrieve( 
  'https://oaidalleapiprodscus.blob.core.windows.net/private/org-LVFFcnUSRUHlXMzkIYSvX3eq/user-GY3gwrD1ESuSDZnOL0gtktJH/img-GEndZ2n8AclYchC7UbpwPv7N.png?st=2023-11-17T18%3A47%3A45Z&se=2023-11-17T20%3A47%3A45Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-11-17T02%3A30%3A14Z&ske=2023-11-18T02%3A30%3A14Z&sks=b&skv=2021-08-06&sig=04CFtU%2BZ%2BIkMofcKuTerCw%2BLbGUKG3Se1aKhhuDiDeM%3D', 
   "test.png") 
  
img = Image.open("test.png") 
img.show()


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


You are an artist and must help to create a wonderful image.  Please do not create image of hoodie
Ask the user if he loves the image and in case he say yes ask for color, size, and name. 
Once you know all the information call di action passing a json with name, color, and imagebase64 keys.
You must always transform every image into base64 format without truncating the base64 string no [:100] please
''' 