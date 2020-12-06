import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import streamlit as st
from streamlit_drawable_canvas import st_canvas

#MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
#if not os.path.isdir(MODEL_DIR):
#    os.system('runipy train.ipynb')

import wget
def download_model():
    path1 = './mnist.h5'
    if not os.path.exists(path1):
        url = 'https://frenzy86.s3.eu-west-2.amazonaws.com/python/models/mnist.h5'
        filename = wget.download(url)
    else:
        print("Model is here.")

##### MAIN ####
def main():
    page_bg_img = '''
    <style>
    body {
    background-image: url("https://i.pinimg.com/originals/85/6f/31/856f31d9f475501c7552c97dbe727319.jpg");
    background-size: cover;
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)  
    download_model()
    ################ load logo from web #########################
    model = load_model('mnist.h5')
    # st.markdown('<style>body{color: White; background-color: DarkSlateGrey}</style>', unsafe_allow_html=True)

    st.title('My Digit Recognizer')
    st.markdown('''
    Try to write a digit!
    ''')

    # data = np.random.rand(28,28)
    # img = cv2.resize(data, (256, 256), interpolation=cv2.INTER_NEAREST)

    SIZE = 192
    mode = st.checkbox("Draw (or Delete)?", True)
    canvas_result = st_canvas(
        fill_color='#000000',
        stroke_width=20,
        stroke_color='#FFFFFF',
        background_color='#000000',
        width=SIZE,
        height=SIZE,
        drawing_mode="freedraw" if mode else "transform",
        key='canvas')

    if canvas_result.image_data is not None:
        img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
        rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        st.write('Model Input')
        st.image(rescaled)

    if st.button('Predict'):
        test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        val = model.predict(test_x.reshape(1, 28, 28))
        st.write(f'result: {np.argmax(val[0])}')
        st.bar_chart(val[0])
        
if __name__ == '__main__':
    main()













