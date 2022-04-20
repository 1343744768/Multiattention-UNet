from PIL import Image
import time
from unet import Unet
import os
unet = Unet()

path = 'img/'
savep = 'save/'
if not os.path.exists(savep):
    os.makedirs(savep)

for i in os.listdir(path):
    img = os.path.join(path, i)
    try:
        image = Image.open(img)
    except:         
        print('Open Error! Try again!')
        continue
    else:
        start = time.time()
        r_image = unet.detect_image(image)
        end = time.time()
        #r_image.show()
        r_image.save(os.path.join(savep, i))
print('finish, time costed:', end - start)
