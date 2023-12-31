#!/bin/python3
from diffusers import StableDiffusionInpaintPipeline
import torch,argparse
from PIL import Image
import requests
from bs4 import BeautifulSoup
from io import BytesIO
import os,datetime

noptions=200
tile_size=150
sd_size=512
allow_nsfw=True
URL="http://enteryourgridurlhere.com"
BASEURL="http://enteryourbaseurlhere.com"
def dummy(images, **kwargs):
    return images, False

def get_image_mask_rows(image):
    fn = lambda x : 255 if x > 254 else 0
    mask = image.convert('L').point(fn, mode='1')
    pmask = Image.new("RGB", (tile_size,tile_size), (0, 0, 0))
    header = {'User-agent' : 'Mozilla/5.0 (Windows; U; Windows NT 5.1; de; rv:1.9.1.5) Gecko/20091102 Firefox/3.5.5'}
    page=requests.get(URL, headers=header).text
    soup = BeautifulSoup(page, 'html.parser')
    rows = soup.find_all('table', attrs={'cellpadding': '0' ,'style':'width:100%; max-width: 450px;display:inline-block;vertical-align:top;'})[0].find_all('tr')
    rowcount=1
    for row in rows:
        data = row.find_all('td')
        for col in range(0,3):
            if len(data[col].find_all('img')) == 1:
                response = requests.get(BASEURL+data[col].find_all('img')[0]['src'])
                part = Image.open(BytesIO(response.content))
                image.paste(part,(col*tile_size,(rowcount-1)*tile_size))
                mask.paste(pmask,(col*tile_size,(rowcount-1)*tile_size))
        rowcount=rowcount+1
    mask = mask.resize((sd_size,sd_size), Image.LANCZOS)
    mask.save('outm.png')
    image.save('outi.png')
    image = image.resize((sd_size,sd_size), Image.LANCZOS)
    return image,mask,rows

def read_arguments():
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-t", "--text",required=True,help="Text that will be used to inpaint the image.")
    argParser.add_argument("-n", "--negative",required=True,help="Negative prompt.")
    argParser.add_argument("-s", "--steps",required=True,type=int,help="Number of steps.")
    argParser.add_argument("-e", "--embed",required=False,help="File to embed.")
    argParser.add_argument("-x", "--embedx",required=False,type=int,default=0,help="Embed x position.")
    argParser.add_argument("-y", "--embedy",required=False,type=int,default=0,help="Embed y position.")
    return argParser.parse_args()

def gen_images(args):
    out_dir=str(datetime.datetime.now().timestamp())
    os.mkdir(out_dir)
    image = Image.new("RGB", (tile_size*3,tile_size*3), (255, 255, 255))
    if(args.embed):
        embed=Image.open(r''+args.embed)
        image.paste(embed,(args.embedx,args.embedy))
    image,mask,rows=get_image_mask_rows(image)
    #pipe = StableDiffusionInpaintPipeline.from_pretrained( "runwayml/stable-diffusion-inpainting",torch_dtype=torch.float16).to('cuda')
    pipe = StableDiffusionInpaintPipeline.from_pretrained( "5w4n/deliberate-v2-inpainting",torch_dtype=torch.float16).to('cuda')
    #pipe = StableDiffusionInpaintPipeline.from_pretrained( "stabilityai/stable-diffusion-2-inpainting",torch_dtype=torch.float16).to('cuda')
    if allow_nsfw:
        pipe.safety_checker = dummy
    for iteration in range(1,noptions):
        torch.cuda.empty_cache()
        outpainted_image = pipe(prompt=args.text, negative_prompt=args.negative,image=image, mask_image=mask,num_inference_steps=args.steps).images[0]
        outpainted_image.save(out_dir+'/'+str(iteration)+'.png')
        os.mkdir(out_dir+'/'+str(iteration))
        resized=outpainted_image.resize((tile_size*3,tile_size*3),Image.LANCZOS)
        rowcount=1
        for row in rows:
            data = row.find_all('td')
            for col in range(0,3):
                if len(data[col].find_all('img')) != 1:
                    outi = resized.crop((col*tile_size,(rowcount-1)*tile_size,(col+1)*tile_size,rowcount*tile_size))
                    outi.save(out_dir+'/'+str(iteration)+'/'+str(col+1)+'-'+str(rowcount)+'.png')
            rowcount=rowcount+1

def main():
    args=read_arguments()
    gen_images(args)

if __name__ == "__main__":
    main()
