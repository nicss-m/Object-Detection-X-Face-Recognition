# import modules
import face_recognition as fr
from math import floor
import numpy as np
import pickle
import queue
import time
import json
import cv2
import os

# import libraries for web development
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase , WebRtcMode, ClientSettings
from typing import List, NamedTuple
from streamlit import caching
from typing import List
from PIL import Image

import streamlit as st
import tempfile
import gdown
import av

# import libraries for service account credentials
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.http import MediaFileUpload
from apiclient.discovery import build

# set page icon and title
favicon = Image.open('favicon.ico')
st.set_page_config(
    page_title="Object Detection X Face Recognition",
    page_icon=favicon,
)

# using Streamlit-Webrtc for real time webcam
WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)

@st.cache
def loading():
    # download model and configuration from google drive
    url_yolo = 'https://drive.google.com/uc?id=1qIcb77gXY0_RV2STYA3tc1ASqyfhfQBv'
    url_yolo_config = 'https://drive.google.com/uc?id=1kn2YYY5ljzg0Z0X1__qdTe3nKX6CG4oR'
    url_coco_names = 'https://drive.google.com/uc?id=1QyYMxO2ER8tR4b3jTlR5JlPxmNjF89jG'
    yolo = "yolov3.weights"
    yolo_config = "yolov3-416.cfg"
    coco_names = "coco.names"
    gdown.download(url_yolo, yolo, quiet=True)
    gdown.download(url_yolo_config, yolo_config, quiet=True)
    gdown.download(url_coco_names, coco_names, quiet=True)
    
    # download pretrained encodings
    url_known_names = 'https://drive.google.com/uc?id=1rqVL7qTdlM5R6fh65Es4ZsaXAuAWx7KS'
    url_encodings = 'https://drive.google.com/uc?id=1qJ2qpouZRemHWiAbsH7P7EXHhpzgk0yt'
    known_names = 'knownNames.txt'
    encodings = 'knownFaces.txt'    
    gdown.download(url_known_names, known_names, quiet=True)    
    gdown.download(url_encodings, encodings, quiet=True)
    
    # object detection variables
    object_names = [] # object names storage
    colors = None # color storage
    net = None # network variable
    
    # face recognition variables
    images = []  # images storage
    knownNames = [] # names storage
    knownEncodeList = [] # known encodes storage

    # extract object_names from coco.names file
    with open(coco_names,'r') as f:
        object_names = f.read().splitlines()
    
    # create random colors (parameters: colors = between 0 and 255 colors, size = number of random colors and channels)    
    colors = np.random.uniform(0,255, size = (len(object_names), 3)) 
    
    # load pretrained encodings (learned faces)
    with open(encodings, "rb") as file:
        knownEncodeList = pickle.load(file)
    with open(known_names, "rb") as file:
        knownNames = pickle.load(file)  
    service = None
    
    return colors,object_names,known_names,encodings,knownEncodeList,knownNames,yolo,yolo_config,service
    
colors,object_names,known_names,encodings,knownEncodeList,knownNames,yolo,yolo_config,service = loading()

def services(service):
    if service!=None:
        pass
    else:
        scope = ['https://www.googleapis.com/auth/drive']
        if os.path.exists('credentials.json'):
            pass
        else:
            CREDENTIALS = st.secrets["CREDENTIALS"]
            CREDENTIALS = json.loads(CREDENTIALS)
            with open('credentials.json','w') as file:
                json.dump(CREDENTIALS,file)

        # parsing JSON credentials for a service account:
        credentials = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
        # create Service
        service = build('drive', 'v3', credentials=credentials)
    return service

service = services(service)

def load_net():
    # create network model using yolov3
    net = cv2.dnn.readNet(yolo,yolo_config)
    return net

def faceRecognition(images):
        
        # resize image to increase running time, producing .25 -> 1/4 scale
        imgResize = cv2.resize(images, (0,0), None, 0.25,0.25)
        
        # convert image to RGB
#         imgRGB = cv2.cvtColor(imgResize, cv2.COLOR_BGR2RGB)
        imgRGB = imgResize
        # extract face locations on each frame and generate encodes of the current faces in the frame
        # it will be use for comparing new faces with the trained faces 
        facesCurFrame = fr.face_locations(imgRGB)
        encodesCurFrame = fr.face_encodings(imgRGB,facesCurFrame)
        
        # name storage
        names = []
        
        # comparing faces through encodings while keeping track of face locations
        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):

            # check the face distance (the lower the value means the more identical the face is)
            faceDistance = fr.face_distance(knownEncodeList, encodeFace)       
            # extract minimum distance
            minDist = min(faceDistance)
            
            # rectangle color
            r_color = (0,0,255)
                
            # check if face distance value is around the acceptable threshold(<0.5)-> this is only a made-up threshold
            if(minDist<0.5):
                
                # extract index of minimum distance
                index = np.argmin(faceDistance)
                
                # x1,y1 -> locations, x2,y2 -> width & height
                y1, x2, y2, x1 = faceLoc
    
                # rescale values, since we resize the image earlier
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

                # extract face name
                name = knownNames[index].upper()
                    
                # draw rectangle in face
                cv2.rectangle(images, (x1,y1), (x2,y2), r_color, 2)

                # draw filled rectangle at the top (emphasize text through background)
                cv2.rectangle(images, (x1,y1), (x2,y1+25), r_color, cv2.FILLED)

                # put text
                cv2.putText(images, name, (x1, y1+20), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2)
                
                names.append(name)
            else:
                
                # x1,y1 -> locations, x2,y2 -> width & height
                y1, x2, y2, x1 = faceLoc
    
                # rescale values, since we resize the image earlier
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

                # unknown name
                name = 'UNKNOWN'

                # draw rectangle in face
                cv2.rectangle(images, (x1,y1), (x2,y2), r_color, 2)

                # draw filled rectangle at the top (emphasize text through background)
                cv2.rectangle(images, (x1,y1), (x2,y1+25), r_color, cv2.FILLED)

                # put text
                cv2.putText(images, name, (x1, y1+20), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2)
            
        return images, names
    
def objectDetection(img,net):     
    
    # resize image to increase running time, producing .25 -> 1/4 scale
    imgResize = cv2.resize(img, (0,0), None, 0.25,0.25)
        
    # obtain height and width
    height, width, _ = imgResize.shape

    # create font (for later use)
    font = cv2.FONT_HERSHEY_PLAIN

    # create blob using blobFromImage function in order to feed the data into the neural network(net)
    blob = cv2.dnn.blobFromImage(imgResize, 1/255, (318, 318), (0,0,0), crop = False)

    # set the Input from the blob into the network
    net.setInput(blob)

    # get the output layers names
    output_layers_names = net.getUnconnectedOutLayersNames()

    # pass the output layers name into the net.forward to get the outputs
    layerOutputs = net.forward(output_layers_names)

    # visualizing results
    boxes = []
    confidences = []
    object_names_ids = []
    
    for output in layerOutputs: # extract the information from the layerOutputs
        for detection in output: # extract the information from each of the outputs

            # note: the first four elements is the locations of the bounding boxes and the fifth element is the confidence score/probability
            # store score predictions. score predictions starts at 6th element or index 5
            scores = detection[5:]
            # extract highest scores using np.argmax
            object_names_id = np.argmax(scores)
            # extract real scores/probality using the extracted highest scores
            confidence = scores[object_names_id]
            # check confidence if larger than 0.3 start to locate boxes
            if confidence > 0.5:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                # get the positions of up and left corner
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                
                # rescale values, since we resize the image earlier
                y, w, h, x = y*4, w*4, h*4, x*4
                
                # append all obtained informations
                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))            
                object_names_ids.append(object_names_id)

    # NMSBoxes: parameters: boxes, obtained confidences, acceptable confidence level, default = 0.4
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    # labels
    label_s = []
    # confidences
    confidence_s = []
    # keeping track of the number of persons detected per frame (determine whether to perform face recognition)
    p_count = 0
        
    # check if indexes is Not Null/ none detected
    if len(indexes)!=0:
        
        for i in indexes.flatten():
            
            # check if there's a person detected (determine whether to perform face recognition)
            if object_names[object_names_ids[i]] == 'person':
                p_count+=1
            # extract x and y coordinates and width height of boxes from boxes
            x, y, w, h = boxes[i]
            # extract label from the object names
            label = str(object_names[object_names_ids[i]]).upper()
            # extract corresponding confidence
            confidence = str(round(confidences[i],2))
            # assign color for boxes
            color = colors[object_names_ids[i]]
            # create a rectange: parameters: img, coordinates, size, color, thickness
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.rectangle(img, (x,y), (x+w, y+20), color, cv2.FILLED)

            # put text: parameters: img, text, location, font, font-size, color, text-thickness
            cv2.putText(img, label + " " + confidence, (x, y+20), font, 1, (255,255,255), 2)
            label_s.append(label)
            confidence_s.append(confidence)
            
    return img,label_s,confidence_s, p_count

def getEncodings(images):

    encodeList = [] # encodes temp storage

    # generate encodings on each images
    for img in images:

        # convert image to RGB, (face recognition only accepts RGB format)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # generate and append encodings
        encode = fr.face_encodings(img_rgb)[0]
        encodeList.append(encode)

    # return encodings
    return encodeList

def Home():
    st.title("Hello Cutie :)")
    st.write("\nWelcome to this for fun app that detects objects and recognizes people faces in images, video or real time camera. Hope you have fun!")
    st.write("Source code is available on my github account: [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/nicss-m/object-detection-x-face-recognition/web.py). Have Fun Learning!")
    st.write("\n\nTerms & Agreement: Any information uploaded or put in this site by the user is of the user will alone and is not responsibility or concern by the creator of this site. By using the site, you therefore agree with the said Terms & Agreement.")
    st.write("\nNote: No actual image or video is saved in this site. All uploaded files are removed after use")

def Learn_Faces():
    
    file_id_names = "1rqVL7qTdlM5R6fh65Es4ZsaXAuAWx7KS"
    file_id_encodings = "1qJ2qpouZRemHWiAbsH7P7EXHhpzgk0yt"
    
    st.title('Learn New Faces')
    st.write('Not yet known? Upload your own image (one person per image only) with the filename as your name or the name of the person in the image.')
    
    multi_files = st.file_uploader("Please upload your selfie image/s", 
                        type = ['jpg','png','jpeg'], accept_multiple_files=True)
    
    st.subheader('For Example:')
    col1, col2 = st.beta_columns(2)
  
    with open('img_rsc/sample2.jpg', 'rb') as f:
        col1.image(np.array(Image.open(f)))
    with open('img_rsc/sample1.jpg', 'rb') as f:
        col2.image(np.array(Image.open(f)))
    
    col1.write('Filename: Lalisa.jpg')
    col2.write('Filename: Han So Hee.jpg')
    images = []
    faceNames = []
    if multi_files != []:
        with st.spinner("Processing... Please wait until the end of process."):
            for file in multi_files:
                if file is not None:
                    # read image
                    img = Image.open(file)
                    img = np.array(img)

                    # extract and append image
                    images.append(img)

                    # extract names without file extension
                    faceNames.append(file.name.split('.')[0])

            # call encoding function to encode/train/learn all faces in the images
            newEncodings =  getEncodings(images)

            # load pretrained encodings (learned faces)
            with open(encodings, "rb") as file:
                knownEncodeList = pickle.load(file)
            with open(known_names, "rb") as file:
                knownNames = pickle.load(file)

            # add new learned faces
            knownEncodeList.extend(newEncodings)
            knownNames.extend(faceNames)

            # create the new files
            with open(encodings, "wb") as file:
                pickle.dump(knownEncodeList, file)
            with open(known_names, "wb") as file:
                pickle.dump(knownNames, file)

            media_content_1 = MediaFileUpload(known_names, mimetype='txt/plain')
            media_content_2 = MediaFileUpload(encodings, mimetype='txt/plain')

            service.files().update(
                fileId=file_id_names,
                media_body=media_content_1
            ).execute()
            
            service.files().update(
                fileId=file_id_encodings,
                media_body=media_content_2
            ).execute()
            
        with st.spinner('Processing done! Rerunning the Program...'):
            time.sleep(3)
            
        multi_files = []
        caching.clear_cache()
    
def ObjD_FaceR_Image():
    net = load_net()
    # image storage
    img = None
    
    st.title('Image Detection')
    multi_files = st.file_uploader("Choose an image/s to detect", 
                            type = ['jpg','png','jpeg'], accept_multiple_files=True)
    if multi_files != []:
        for file in multi_files:    
            if file is not None:
                # read image
                img = Image.open(file).convert("RGB")
                orig_img = np.array(img)
                img = orig_img.copy()
                
                # perform object detection
                img,_,_,p_count = objectDetection(img,net)
                
                if p_count>0:
                    # perform face recognition
                    img,_ = faceRecognition(img)
                
                # visualize
                st.write("Original Image:")
                st.image(orig_img, use_column_width=True)
                st.write("Result:")
                st.image(img, use_column_width=True)
    
    return img

def ObjD_FaceR_Video():
    net = load_net()
    st.title('Video Detection')
    file = st.file_uploader("Choose a video to detect", type = ['mp4'])
    stframe = st.empty()

    if file is not None:
        temp = tempfile.NamedTemporaryFile(delete=False) 
        temp.write(file.read())
        cap = cv2.VideoCapture(temp.name)
        
        # for video result
        result_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.webm')

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        size = (frame_width, frame_height) 
        result = cv2.VideoWriter(result_temp.name, cv2.VideoWriter_fourcc(*'VP90'), 20, size)
        
        with st.spinner("Processing... Please wait until the end of process."):
            # progress bar
            bar = st.progress(0)
            length,add,count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),0,0
            if length != 0:
                add = 100/length
            else:
                add = 0
                
            while True:
                _, img = cap.read()
                # check if null value
                if np.shape(img) != ():
                    
                    # perform object detection
                    img,_,_,p_count = objectDetection(img,net)
                    
                    if p_count>0:    
                        # perform face recognition
                        img,_ = faceRecognition(img)
                    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    
                    stframe.image(img)

                    # saving video frames
                    result.write(img)

                    # update progress bar
                    count+=add
                    bar.progress(floor(count))

                else:
                    bar.empty()
                    break
        stframe.empty()
        with st.spinner('Processing done! Please wait for the result...'):
            # output video
            cap.release()
            result.release()
            st.title("Result")
            with open(result_temp.name, 'rb') as f:
                st.video(f)

        
def  ObjD_FaceR_Cam():
    net = load_net()
    st.title('Camera Detection')
    st.markdown("Click the start button to start capturing")
    
    result_queue: "queue.Queue[List[Detection]]"
        
    class Detection(NamedTuple):
        Label: str
        Score: str
        Face: str
            
    class VideoTransformer(VideoTransformerBase):
        def __init__(self)-> None:
            self.result_queue = queue.Queue()
            
        def _annotate_image(self, img):
            result: List[Detection] = []
            names = []    
            
            # perform object detection
            img,labels,scores,p_count = objectDetection(img,net)
            
            if p_count>0:
                # perform face recognition
                img,names = faceRecognition(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 
            
            # append results
            result.append(Detection(Label=labels, Score=scores, Face=names))
                
            return img, result
        
        def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_ndarray(format="rgb24")
            blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),(104.0, 177.0, 123.0))
            output_img, result = self._annotate_image(image)
            
            self.result_queue.put(result)
            return output_img
                
    webrtc_ctx = webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV,
                        client_settings=WEBRTC_CLIENT_SETTINGS,video_transformer_factory=VideoTransformer,async_transform=True)
    
    if st.checkbox('Show detected labels', value = True):
        if webrtc_ctx.state.playing:
            labels_placeholder = st.empty()
            while True:
                if webrtc_ctx.video_processor:
                    try:
                        result = webrtc_ctx.video_processor.result_queue.get(timeout=1.0)
                    except queue.Empty:
                        result = None
                    labels_placeholder.table(result)
                else:
                    break
                    
def main():
    st.sidebar.title("Menu")
    selected_box = st.sidebar.radio(
    'Go to',
    ('Home','Learn Faces','Image', 'Video', 'Camera'))
    
    if selected_box == 'Home':
        Home() 
    if selected_box == 'Learn Faces':
        Learn_Faces() 
    if selected_box == 'Image':
        ObjD_FaceR_Image()
    if selected_box == 'Video':
        ObjD_FaceR_Video()
    if selected_box == 'Camera':
        ObjD_FaceR_Cam()

# Run web app
if __name__ == "__main__":
    main()
