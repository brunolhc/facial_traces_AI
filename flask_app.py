from flask import Flask, render_template, request, session, send_file
import cv2
import os
import uuid
import numpy as np
from copy import deepcopy
import scipy
import tensorflow as tf

###############################################################################################################################################

class EYECOLOR():
  def __init__(self):
    self.mobilenetv2_Unet = tf.keras.models.load_model('./models/mobilenet_unet.h5', compile=False)
    self.size_image_model = (256,256)
    self.num_clusters = 6

  def __call__(self, inp_image):
    inp_image, eye_rgb, eye_orig, frequency_percent, result = self.eye_position_n_color(inp_image,
                                                          is_bgr = False)

    return inp_image, eye_rgb, eye_orig, frequency_percent, result

################################################################################
  def call_mobilenetv2_Unet(self, image):
################################################################################
    tol = 0.5
    image_np = deepcopy(image)

    image_np = cv2.resize(image_np,
                          self.size_image_model,
                          interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(np.uint8(image_np),
                        cv2.COLOR_BGR2GRAY)

    image_np[:,:,1] = gray
    image_np[:,:,2] = gray
    image_np = np.array(image_np).astype(np.float32)
    image_np = image_np.reshape(self.size_image_model+tuple([3]))
    image_np = np.expand_dims(image_np, 0)

    mask_out = self.mobilenetv2_Unet.predict(image_np)
    mask_out = np.asarray(mask_out, np.float32)
    mask_out = np.squeeze(np.asarray(mask_out))

    mask = np.zeros(self.size_image_model)
    mask[np.where(mask_out > tol)] = 255

    return image_np, mask

################################################################################
  def eye_position_n_color(self, inp_image, is_bgr, full_eye = False):
################################################################################
    image_height, image_width, channels = inp_image.shape

    center_e, radius_e = self.iris_refination(inp_image)
    radius_e = 0.8*np.array(radius_e)
    if is_bgr:
      inp_image = cv2.cvtColor(inp_image, cv2.COLOR_BGR2RGB)

    mask_lab = np.zeros([image_height, image_width, channels])
    mask_rgb = np.zeros([image_height, image_width, channels])


    for x in range(image_width):
      for y in range(image_height):
          distance = np.sqrt((x - center_e[0][0])**2 + (y - center_e[0][1])**2)
          if distance <= radius_e[0]:
              color = self.human_iris_color_map(inp_image[y, x, :], False)
              mask_lab[y, x, :] = color[1]
              mask_rgb[y, x, :] = color[2]

    eye = mask_lab[
        int(center_e[0][1]-full_eye*radius_e[0]):int(center_e[0][1]+radius_e[0]),
        int(center_e[0][0]-radius_e[0]):int(center_e[0][0]+radius_e[0]),
        :
        ]

    eye_rgb = mask_rgb[
        int(center_e[0][1]-full_eye*radius_e[0]):int(center_e[0][1]+radius_e[0]),
        int(center_e[0][0]-radius_e[0]):int(center_e[0][0]+radius_e[0]),
        :
        ]

    eye_orig = deepcopy(inp_image[
        int(center_e[0][1]-full_eye*radius_e[0]):int(center_e[0][1]+radius_e[0]),
        int(center_e[0][0]-radius_e[0]):int(center_e[0][0]+radius_e[0]),
        :
        ])

    eye_rgb = np.asarray(eye_rgb,
                         np.int32)
    shape = eye.shape
    NUM_CLUSTERS = self.num_clusters
    pixels_inside_contour = eye.reshape(np.prod(shape[:2]),
                                        shape[2]).astype(float)

    codes, dist = scipy.cluster.vq.kmeans(pixels_inside_contour,
                                          NUM_CLUSTERS)


    vecs, dist = scipy.cluster.vq.vq(pixels_inside_contour,
                                     codes)
    counts, bins = np.histogram(vecs,
                                len(codes))

    index_sorted = np.argsort(counts)[::-1]
    counts = counts[index_sorted]
    codes = codes[index_sorted]

    frequency_percent = []
    results = []

    total = 0
    for i in range(len(codes)):
      cm = self.human_iris_color_map(codes[i], True)
      if cm[0] != 'Black':
        total += counts[i]

    for i in range(len(codes)):
      cm = self.human_iris_color_map(codes[i], True)
      if cm[0] != 'Black':
        frequency_percent.append((cm[0], cm[2], "{:.2f}".format(counts[i]/total)))

    f_size = 0.001*image_width
    text_t = int(f_size)

    pixel_x = int(center_e[0][0] + 0.6*radius_e[0]*np.cos(np.deg2rad(180)))
    pixel_y = int(center_e[0][1] + 0.6*radius_e[0]*np.sin(np.deg2rad(180)))

    border_h = [int(0.02*image_height), int(0.02*image_height)]
    border_w = [int(0.005*image_width), int(0.01*image_width)]

    inp_image = cv2.line(inp_image,
                         (int(pixel_x), int(pixel_y)),
                         (int(pixel_x), int(pixel_y+0.1*image_height)),
                         (0,0,0),
                         thickness=1)

    for i in codes:
      text = self.human_iris_color_map(i, True)
      rgb_color = np.array(text[2]).astype(int)
      if text[0] != 'Black':
        break


    #rgb_color = np.array(codes[0]).astype(int)#np.array(inp_image[pixel_x][pixel_y]).astype(int)

    index = text[0].find(' ')

    if index != -1:
      word = text[0].split(' ')[1]
    else:
      word = text[0]

    size, _ = cv2.getTextSize(word,
                              cv2.FONT_HERSHEY_DUPLEX,
                              f_size,
                              text_t)

    org = (int(pixel_x + 4), int(pixel_y + 0.1*image_height - size[1]/2))

    start_point = [int(org[0] - border_w[0]),
                   int(org[1] - size[1]/2 - border_h[0])]

    end_point =  [start_point[0]+int(size[0])+border_w[1],
                  start_point[1]+int(size[1])+border_h[1]]

    inp_image = cv2.rectangle(inp_image,
                              start_point,
                              end_point,
                              color = rgb_color.tolist(),
                              thickness = -1)

    inp_image = cv2.putText(inp_image, text=word,
                            org=org,
                            fontFace=cv2.FONT_HERSHEY_DUPLEX,
                            fontScale=f_size,
                            color=(255,255,255),
                            thickness=text_t)
    result = []
    i=0
    for l in frequency_percent:
      result.append(str(int(100*float(l[2]))) + '%' + ' ' + l[0])
      i += 1 
    
    return inp_image, eye_rgb, eye_orig, frequency_percent, result

################################################################################
  def rgb_to_lab(self, inputColor):
################################################################################
   num = 0
   RGB = [0, 0, 0]

   for value in inputColor :
       value = float(value) / 255

       if value > 0.04045 :
           value = ( ( value + 0.055 ) / 1.055 ) ** 2.4
       else :
           value = value / 12.92

       RGB[num] = value * 100
       num = num + 1

   XYZ = [0, 0, 0,]

   X = RGB [0] * 0.4124 + RGB [1] * 0.3576 + RGB [2] * 0.1805
   Y = RGB [0] * 0.2126 + RGB [1] * 0.7152 + RGB [2] * 0.0722
   Z = RGB [0] * 0.0193 + RGB [1] * 0.1192 + RGB [2] * 0.9505
   XYZ[ 0 ] = round( X, 4 )
   XYZ[ 1 ] = round( Y, 4 )
   XYZ[ 2 ] = round( Z, 4 )

   # ref_X =  95.047   Observer= 2°, Illuminant= D65
   XYZ[ 0 ] = float( XYZ[ 0 ] ) / 95.047
   XYZ[ 1 ] = float( XYZ[ 1 ] ) / 100.0          # ref_Y = 100.000
   XYZ[ 2 ] = float( XYZ[ 2 ] ) / 108.883        # ref_Z = 108.883

   num = 0
   for value in XYZ :

       if value > 0.008856 :
           value = value ** ( 0.3333333333333333 )
       else :
           value = ( 7.787 * value ) + ( 16 / 116 )

       XYZ[num] = value
       num = num + 1

   Lab = [0, 0, 0]

   L = ( 116 * XYZ[ 1 ] ) - 16
   a = 500 * ( XYZ[ 0 ] - XYZ[ 1 ] )
   b = 200 * ( XYZ[ 1 ] - XYZ[ 2 ] )

   Lab [ 0 ] = round( L, 4 )
   Lab [ 1 ] = round( a, 4 )
   Lab [ 2 ] = round( b, 4 )

   return Lab

################################################################################
  def human_iris_color_map(self, color, is_lab):
################################################################################
    iris = {'Black' : [(0,0,0), (0, 0, 0)],
    'Brown' : [(35, 10, 10), (102, 76, 67)],
    'Light Brown' : [(50, 20, 20), (158, 105, 86)],
    'Blue' : [(50, -30, -50), (0, 15, 204)], #BlueA
    'Light Blue' : [(70, -20, -40), (61, 184, 243)], #BlueB
    'Green' :  [(40, -40, 10), (0, 110, 76)], #GreenA
    'Light Green' :  [(60, -30, 30), (101, 158, 89)], #Green B
    'Hazel' : [(45, -20, -20), (29, 117, 139)], #HazelA
    'Light Hazel' : [(65, 10, 20), (185, 151, 122)], #HazelB
    'Gray' : [(35,0,0), (82, 82, 82)], #GrayA
    'Light Gray' : [(55,5,5), (143, 129, 123)], #GrayB
    'Amber': [(50, 20, 40), (163, 105, 50)],
    'Light Amber': [(70, 30, 50), (237, 149, 80)],
    'Red':[(30, 50, 40), (141, 13, 10)],
    'Light Red':[(40, 60, 50), (183, 24, 13)],
    'Violet': [(25, 40, -50), (83, 31, 137)],
    'Light Violet': [(35, 50, -40), (132, 41, 148)]}

    if is_lab:
      lab_color = color
    else:
      lab_color = self.rgb_to_lab(np.array(color))

    dist_min = 999999
    color_name = 'unknow'
    color_value = (0,0,0)
    for key, value in iris.items():
      dist = np.linalg.norm(np.array(value[0]) - np.array(lab_color))

      if dist<dist_min:
        dist_min = dist
        color_name = key
        lab_value = value[0]
        rgb_value = value[1]

    return color_name, lab_value, rgb_value

################################################################################
  def iris_refination(self, image):
################################################################################
    image_height, image_width, _ = image.shape
    _, mask = self.call_mobilenetv2_Unet(image)
    gray = np.uint8(np.squeeze(mask[:,:]))

    canny = cv2.Canny(gray, 30, 200)

    contours, _ = cv2.findContours(canny,
                                  cv2.RETR_LIST,
                                  cv2.CHAIN_APPROX_SIMPLE)

    sorted_contours = sorted(contours,
                             key=lambda x: cv2.contourArea(x),
                             reverse=True)

    image_contours = deepcopy(image)
    mask_px = np.zeros_like(image[:,:,0])

    contour_e1 = []
    pixels_inside_contour = []
    for contour in sorted_contours[2:3]:
      for point in contour:
        point[0][0] = int(point[0][0]*image_width/self.size_image_model[0])
        point[0][1] = int(point[0][1]*image_height/self.size_image_model[1])
        contour_e1.append(point)

    contour_e2 = []
    for contour in sorted_contours[4:5]:
      for point in contour:
        point[0][0] = int(point[0][0]*image_width/self.size_image_model[0])
        point[0][1] = int(point[0][1]*image_height/self.size_image_model[1])
        contour_e2.append(point)

    contour_e = [np.array(contour_e1), np.array(contour_e2)]

    center_e = []
    radius_e = []
    for contour in contour_e:
      M = cv2.moments(contour)
      if M['m00'] != 0:
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
      else:
        cX, cY = 0, 0
      center = (int(cX), int(cY))
      center_e.append(center)

      total_radius, count = 0,0
      for point in contour:
          (x, y) = point[0]
          distance = np.sqrt((x - center[0])**2 + (y - center[1])**2)
          if distance >= 0:
              total_radius += distance
              count += 1
      if count > 0:
        mean_radius = total_radius / count
        radius_e.append(mean_radius)

    if radius_e == []:
      raise Exception("Face location fail.")
      

    return center_e, radius_e

################################################################################
  def facial_contour(self, inp_image):
################################################################################
    _, mask = self.call_mobilenetv2_Unet(inp_image)
    image_height, image_width, _ = inp_image.shape

    gray = np.uint8(np.squeeze(mask[:,:]))

  # Apply thresholding or other preprocessing techniques if necessary
    canny = cv2.Canny(gray, 30, 200)

  # Find contours in the image
    contours, _ = cv2.findContours(canny,
                                  cv2.RETR_LIST,
                                  cv2.CHAIN_APPROX_SIMPLE)

    sorted_contours = sorted(contours,
                             key=lambda x: cv2.contourArea(x),
                             reverse=True)

    image_contours = deepcopy(inp_image)
    # to isolete the eyes from the face, we can remove the biggest contour
    for contour in sorted_contours:
      for point in contour:
        #print(point[0][1])
        point[0][0] = int(point[0][0]*image_width/256)
        point[0][1] = int(point[0][1]*image_height/256)

      #cv2.drawContours(image_contours, [contour], -1, (0, 255, 0), int(image_width/400))

    for contour in sorted_contours[:1]:
      cv2.drawContours(image_contours, [sorted_contours[0]], -1, (0, 255, 0), int(image_width/300))


    return(image_contours)

################################################################################
  def face_bbox(self, inp_image):
################################################################################
    image, mask = self.call_mobilenetv2_Unet(inp_image)

    mask_indexes = np.where(mask[:,:] > 200)

    face_xmin = np.min(mask_indexes[1])
    face_xmax = np.max(mask_indexes[1])
    face_ymin = np.min(mask_indexes[0])
    face_ymax = np.max(mask_indexes[0])

    return image, face_xmin/256, face_ymin/256, face_xmax/256, face_ymax/256

################################################################################
  def eye_bbox(self, inp_image):
################################################################################
    image, mask = self.call_mobilenetv2_Unet(inp_image)
    image_height, image_width, _ = inp_image.shape

    gray = np.uint8(np.squeeze(mask[:,:]))

  # Apply thresholding or other preprocessing techniques if necessary
    canny = cv2.Canny(gray, 30, 200)

  # Find contours in the image
    contours, _ = cv2.findContours(canny,
                                  cv2.RETR_TREE,
                                  cv2.CHAIN_APPROX_SIMPLE)

    sorted_contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    image_contours = deepcopy(image)

    # first eye
    e1_xmin, e2_xmin = 1, 1
    e1_xmax, e2_xmax = 0, 0
    e1_ymin, e2_ymin = 1, 1
    e1_ymax, e2_ymax = 0, 0

    for contour in sorted_contours[2:3]:

      for point in contour:
        #print(point[0][1])
        x = point[0][0]/256
        y = point[0][1]/256

        if x < e1_xmin:
          e1_xmin = x
        if x > e1_xmax:
          e1_xmax = x
        if y < e1_ymin:
          e1_ymin = y
        if y > e1_ymax:
          e1_ymax = y
        #new_contour.append(point)

    for contour in sorted_contours[4:5]:
      new_contour = []
      image_contours = deepcopy(image)
      for point in contour:
        x = point[0][0]/256
        y = point[0][1]/256

        if x < e2_xmin:
          e2_xmin = x
        if x > e2_xmax:
          e2_xmax = x
        if y < e2_ymin:
          e2_ymin = y
        if y > e2_ymax:
          e2_ymax = y

    return image, 0.88*e1_xmin, 1.12*e1_xmax, 0.97*e1_ymin, 1.03*e1_ymax,\
                0.92*e2_xmin, 1.08*e2_xmax, 0.97*e2_ymin, 1.03*e2_ymax

################################################################################
  def cut_face_n_eyes(self, inp_image):
################################################################################
    image_height, image_width, _ = inp_image.shape

    image, x, y, x_max, y_max = self.face_bbox(inp_image)

    image, e1_xmin, e1_xmax, e1_ymin, e1_ymax, e2_xmin, e2_xmax, e2_ymin, e2_ymax = self.eye_bbox(inp_image)

    face = deepcopy(inp_image[ 
                    int(y*image_height):int(y_max*image_height),
                    int(x*image_width):int(x_max*image_width),
                    :])
    eye1 = deepcopy(inp_image[ 
                    int(e1_ymin*image_height):int(e1_ymax*image_height),
                    int(e1_xmin*image_width):int(e1_xmax*image_width),
                    :])
    eye2 = deepcopy(inp_image[ 
                    int(e2_ymin*image_height):int(e2_ymax*image_height),
                    int(e2_xmin*image_width):int(e2_xmax*image_width),
                    :])
    image_bbox = cv2.rectangle(inp_image, pt1 = (int(x*image_width), int(y*image_height)),
                                   pt2 = (int(x_max*image_width), int(y_max*image_height)),
                                   color = (0,0,255), thickness = 2)
    image_bbox = cv2.rectangle(image_bbox, pt1 = (int(e1_xmin*image_width), int(e1_ymin*image_height)),
                                   pt2 = (int(e1_xmax*image_width), int(e1_ymax*image_height)),
                                   color = (0,0,255), thickness = 2)
    image_bbox = cv2.rectangle(image_bbox, pt1 = (int(e2_xmin*image_width), int(e2_ymin*image_height)),
                                   pt2 = (int(e2_xmax*image_width), int(e2_ymax*image_height)),
                                   color = (0,0,255), thickness = 2)

    return image_bbox, face, eye1, eye2
#########################################################################################################
  
global iris_map, facial_contour, cut_face, cut_eye 
iris_map=0
facial_contour=0
cut_face=0
cut_eye=0


app = Flask(__name__, template_folder='./template', static_folder = './static')
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'

eye_color = EYECOLOR()

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST','GET']) 
def upload():
  #if request.method == 'POST':  
    if 'image' not in request.files:
        return 'No image uploaded!', 400

    uploaded_image = request.files['image']
    #filename = str(uuid.uuid4()) + '.jpg'  # Unique filename for the image

    if uploaded_image.filename == '':
        return 'No selected file'

    # Save the uploaded image to the server's filesystem
    imagepath = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4()) + '.jpg')
    eyepath = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4()) + '.jpg')
    mappath = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4()) + '.jpg')
    uploaded_image.save(imagepath)

    # Store the image filename in the session
    #session['image_filename'] = filename
    session['uploaded_img_file_path'] = imagepath
    session['eye_path'] = eyepath
    session['map_path'] = mappath
    #filepath = session.get('uploaded_img_file_path', None)

    bgr_image = cv2.imread(imagepath)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    rgb_image, eye_map, eye_orig, frequency_percent, result = eye_color(rgb_image)
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    cv2.imwrite(imagepath, bgr_image)
    cv2.imwrite(eyepath, cv2.cvtColor(eye_orig, cv2.COLOR_RGB2BGR))
    eye_map = eye_map.astype(np.uint8)
    cv2.imwrite(mappath, cv2.cvtColor(eye_map, cv2.COLOR_RGB2BGR))

    return render_template('display.html', result = result)

@app.route('/requests',methods=['POST','GET'])
def tasks():
  if request.method == 'POST':
    if 'image' not in request.files:
        return 'No image uploaded!', 400

    uploaded_image = request.files['image']

    if uploaded_image.filename == '':
          return 'No selected file'

    # Save the uploaded image to the server's filesystem
    imagepath = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4()) + '.jpg')
    uploaded_image.save(imagepath)

    # Store the image filename in the session
    session['uploaded_img_file_path'] = imagepath
    
    

    if request.form.get('iris_map') == 'Iris Color Mapping':
      eyepath = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4()) + '.jpg')
      mappath = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4()) + '.jpg')

      # Store the image filename in the session
      session['uploaded_img_file_path'] = imagepath
      session['eye_path'] = eyepath
      session['map_path'] = mappath

      bgr_image = cv2.imread(imagepath)
      rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
      rgb_image, eye_map, eye_orig, frequency_percent, result = eye_color(rgb_image)
      bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

      cv2.imwrite(imagepath, bgr_image)
      cv2.imwrite(eyepath, cv2.cvtColor(eye_orig, cv2.COLOR_RGB2BGR))
      eye_map = eye_map.astype(np.uint8)
      cv2.imwrite(mappath, cv2.cvtColor(eye_map, cv2.COLOR_RGB2BGR))

      return render_template('eye_map_res.html', result = result)
    
    elif request.form.get('facial_contours') == 'Facial Contour':
      bgr_image = cv2.imread(imagepath)
      rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
      rgb_image = eye_color.facial_contour(rgb_image)
      bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
      cv2.imwrite(imagepath, bgr_image)
      
      return render_template('facial_contour_res.html')
  
    elif  request.form.get('cut_face') == 'Cut Face': 
      eyepath = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4()) + '.jpg')

      # Store the image filename in the session
      session['uploaded_img_file_path'] = imagepath
      session['eye_path'] = eyepath

      bgr_image = cv2.imread(imagepath)
      rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
      rgb_image, face, _, _ = eye_color.cut_face_n_eyes(rgb_image)
      bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

      cv2.imwrite(imagepath, bgr_image)
      cv2.imwrite(eyepath, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
  
      return render_template('face_cut_res.html')
    
    elif  request.form.get('cut_eyes') == 'Cut Eyes':
      eyepath = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4()) + '.jpg')
      mappath = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4()) + '.jpg')

      # Store the image filename in the session
      session['eye_path'] = eyepath
      session['map_path'] = mappath

      bgr_image = cv2.imread(imagepath)
      rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
      rgb_image, _, eye1, eye2 = eye_color.cut_face_n_eyes(rgb_image)
      bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

      cv2.imwrite(imagepath, bgr_image)
      cv2.imwrite(eyepath, cv2.cvtColor(eye1, cv2.COLOR_RGB2BGR))
      cv2.imwrite(mappath, cv2.cvtColor(eye2, cv2.COLOR_RGB2BGR))

      return render_template('eye_cut_res.html')
    
        
@app.route('/display')
def display():
    if 'image_filename' not in session:
        return 'No image found!', 404

    # Retrieve the stored image filename from the session
    filename = session['image_filename']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Read and process the image using OpenCV
    bgr_image = cv2.imread(filepath)

    return 'Image displayed successfully!'

@app.route('/get_image')
def get_image():
    # Retrieve the stored image filename from the session
    filepath = session['uploaded_img_file_path']

    # Send the image file to the client for display
    return send_file(filepath, mimetype='image/jpeg')

@app.route('/get_eye')
def get_eye():
    # Retrieve the stored image filename from the session
    filepath = session['eye_path']

    # Send the image file to the client for display
    return send_file(filepath, mimetype='image/jpeg')

@app.route('/get_map')
def get_map():
    # Retrieve the stored image filename from the session
    filepath = session['map_path']

    # Send the image file to the client for display
    return send_file(filepath, mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(debug=True)
