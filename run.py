import fastai
from fastai.vision import *
from fastai.callbacks import *
from fastai.utils.mem import *
from torchvision.models import vgg16_bn
from fastai.vision.gan import *
import os, urllib
import cv2
import numpy as np
import pandas as pd

import streamlit as st
import skimage

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

path=Path()
path_pic=path/'image'
path_picbw=path/'image/image_bw'
src = ImageImageList.from_folder(path_picbw).split_by_rand_pct(0.1, seed=42)

example_dic={'I wish to upload my own photo':'',
    'Broadway at the United States Hotel, Saratoga Springs, N.Y.':'4a27640r.jpg',
    'Nat. Am. Ballet, 8/20/24':'colourized-black-and-white-photography-history-1-2.jpg',
    'Rosalie Jones starting on campaign tour for LaFollette, Sept. 24':'3c15477r',
    'Boxing':'service-pnp-hec-29200-29201v.jpg',
    'Umpire watches as New York Yankee player slides into base ahead of the tag during baseball game with Washington':'3c35415r.jpg',
    'Country store on dirt road.':'8b33922r.jpg',
    'Portrait of Louis Armstrong':'iiif-service_music_musgottlieb_musgottlieb-00201_ver01_0001-full-pct_25.0-0-default.jpg',
    'Antietam, Md. President Lincoln and Gen. George B. McClellan':'04351r.jpg',
    'Bing Miller, of the Philadelphia Athletics, tagged out at home plate by Washington Nationals catcher "Muddy" Ruel during baseball game':'3c35437r.jpg',
    'Buying Easter flowers, Union Sq. [New York]':'00294r.jpg'}

def main():
    for filename in EXTERNAL_DEPENDENCIES.keys():
        download_file(filename)
    st.title("Black&White Photos Colorisation")
    example=st.selectbox('Try examples:',['I wish to upload my own photo',
    'Broadway at the United States Hotel, Saratoga Springs, N.Y.',
    'Nat. Am. Ballet, 8/20/24',
    'Rosalie Jones starting on campaign tour for LaFollette, Sept. 24',
    'Boxing',
    'Umpire watches as New York Yankee player slides into base ahead of the tag during baseball game with Washington',
    'Country store on dirt road.',
    'Portrait of Louis Armstrong',
    'Antietam, Md. President Lincoln and Gen. George B. McClellan',
    'Bing Miller, of the Philadelphia Athletics, tagged out at home plate by Washington Nationals catcher "Muddy" Ruel during baseball game',
    'Buying Easter flowers, Union Sq. [New York]'])

    if example == 'I wish to upload my own photo':
        st.set_option('deprecation.showfileUploaderEncoding', False)
        uploaded_file = st.file_uploader(
            "upload a black&white photo", type=['jpg','png','jpeg'])

        if uploaded_file is not None:
            g = io.BytesIO(uploaded_file.read())  # BytesIO Object
            temporary_location = "image/temp.jpg"

            with open(temporary_location, 'wb') as out:  # Open temporary file as bytes
                out.write(g.read())  # Read bytes into file

            # close file
            out.close()
        
            bw_one("image/temp.jpg",img_size=800)
            st.image("image/temp.jpg",width=512)

        start_analyse_file = st.button('Analyse uploaded file')
        if start_analyse_file== True:
            
            learn_gen=create_learner(path='',file='export_5.pkl')
            predict_img("image/image_bw/temp_bw.jpg",learn_gen,img_width=512)
    else:
        bw_one("test_img/"+example_dic[example],img_size=800)
        st.image("test_img/"+example_dic[example],width=512)
        start_analyse_file = st.button('Analyse example')
        if start_analyse_file== True:
            
            learn_gen=create_learner(path='',file='export_5.pkl')
            predict_img("image/image_bw/temp_bw.jpg",learn_gen,img_width=512)

@st.cache(allow_output_mutation=True)
def create_learner(path='model',file='export_3.pkl'):
    learn_gen=load_learner(path='model',file='export_3.pkl')
    return learn_gen

#@st.cache
def predict_img(fn,learn_gen,img_width=640):

    img_test_bw=cv2.imread(fn)
    height,width=img_test_bw.shape[0],img_test_bw.shape[1]
    size=(height,width)
    st.text(size)
    try:
        data_gen = get_data2(1,size)
        learn_gen.data=data_gen
        _,img,b=learn_gen.predict(open_image(fn))
        img_np=image2np(img)
        st.image(img_np,clamp=True,width=img_width)

    except:

        try:
      
            size=(int(height),int(width)+1)
            print(size)
            data_gen = get_data2(1,size)
            learn_gen.data=data_gen
            _,img,b=learn_gen.predict(open_image(fn))
            img_np=image2np(img)
            st.image(img_np,clamp=True,width=img_width)
        except:
            try:
                size=(int(height+1),int(width))
                print(size)
                data_gen = get_data2(1,size)
                learn_gen.data=data_gen
                _,img,b=learn_gen.predict(open_image(fn))
                img_np=image2np(img)
                st.image(img_np,clamp=True,width=img_width)
            except:
                try:
                    size=(int(height+1),int(width+1))
                    print(size)
                    data_gen = get_data2(1,size)
                    learn_gen.data=data_gen
                    _,img,b=learn_gen.predict(open_image(fn))
                    img_np=image2np(img)
                    st.image(img_np,clamp=True,width=img_width)
                except:
                    print(height,width)
                    print('dimension error, resize to square')
                    data_gen = get_data2(1,int(max(height,width)))
                    learn_gen.data=data_gen
                    _,img,b=learn_gen.predict(open_image(fn))
                    img_np=image2np(img)
                    st.image(img_np,clamp=True,width=img_width)



def bw_one(fn,img_size=512):
    dest = 'image/image_bw/temp_bw.jpg'

    # Load the image
    img=cv2.imread(str(fn))
    height,width  = img.shape[0],img.shape[1]

    if max(width, height)>img_size:
        if height > width:
            width=width*(img_size/height)
            height=img_size
            img=cv2.resize(img,(int(width), int(height)))
        elif height <= width:      
            height=height*(img_size/width)
            width=img_size
            img=cv2.resize(img,(int(width), int(height)))

    # Add salt-and-pepper noise to the image
    noise = skimage.util.random_noise(img, mode='gaussian',mean=0,var=0.0025)

    # The above function returns a floating-point image in the range [0, 1]
    # so need to change it to 'uint8' with range [0,255]
    noise = np.array(255 * noise, dtype=np.uint8)
    gray = cv2.cvtColor(noise, cv2.COLOR_BGR2GRAY)
    gray_3 = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(dest),gray_3)



def get_data2(bs,size):
    data = (src.label_from_func(lambda x: path_pic/x.name)
           .transform(size=size)
           .databunch(bs=bs).normalize(imagenet_stats, do_y=True))
    data.c = 3
    return data






# This file downloader demonstrates Streamlit animation.
def download_file(file_path):
    # Don't download the file twice. (If possible, verify the download using the file length.)
    if os.path.exists(file_path):
        if "size" not in EXTERNAL_DEPENDENCIES[file_path]:
            return
        elif os.path.getsize(file_path) == EXTERNAL_DEPENDENCIES[file_path]["size"]:
            return

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % file_path)
        progress_bar = st.progress(0)
        with open(file_path, "wb") as output_file:
            with urllib.request.urlopen(EXTERNAL_DEPENDENCIES[file_path]["url"]) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning("Downloading %s... (%6.2f/%6.2f MB)" %
                        (file_path, counter / MEGABYTES, length / MEGABYTES))
                    progress_bar.progress(min(counter / length, 1.0))

    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()

EXTERNAL_DEPENDENCIES = {
    "export_5.pkl": {
        "url": "https://dl.dropboxusercontent.com/s/u1vozd9ikuryb8d/export_5.pkl?dl=0",
        "size": 246698366
    }
}


if __name__ == "__main__":
    main()
