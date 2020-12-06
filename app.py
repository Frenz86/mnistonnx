import os
import numpy as np
import cv2
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import json
import onnxruntime
import wget

#MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
#if not os.path.isdir(MODEL_DIR):
#    os.system('runipy train.ipynb')

import wget
def download_model():
    path1 = './mnist.onnx'
    if not os.path.exists(path1):
        url = 'https://frenzy86.s3.eu-west-2.amazonaws.com/python/models/mnist.onnx'
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
    # st.markdown('<style>body{color: White; background-color: DarkSlateGrey}</style>', unsafe_allow_html=True)

    st.title('Riconoscitore di numero')
    st.markdown('''
    Disegna un numero!
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
        session = onnxruntime.InferenceSession("mnist.onnx")
        test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        test_x.resize((1, 28, 28,1))
        data = json.dumps({'data': test_x.tolist()})
        data = np.array(json.loads(data)['data']).astype('float32')
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        result = session.run([output_name], {input_name: data})
        prediction=int(np.argmax(np.array(result).squeeze(), axis=0))
        st.write(f'result: {prediction}')
        st.bar_chart(np.array(result).squeeze())
        
if __name__ == '__main__':
    main()













