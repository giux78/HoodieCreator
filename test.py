
from PIL import Image 
  
img1 = Image.open(r"./data/hoodie-black-retro.png") 
img2 = Image.open(r"./data/lynx.png") 

img3 = img2.resize((300,300))
  
# No transparency mask specified,  
# simulating an raster overlay 
img1.paste(img3, (250,280)) 
  
img1.show() 