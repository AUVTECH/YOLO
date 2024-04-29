# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 19:56:58 2022

@author: abdrh
"""


import cv2
import numpy as np

cap = cv2.VideoCapture(0) # buraya 0,1 kamera hatırlatmaa....

while(True):
    ret, frame = cap.read()
    
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]



    frame_blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False) #giriş görüntüsü sinir ağı için Blob haline çevrilir.
# ikinci parametre YOLO yazarlarının önerdiği orandır.
# 3. parametredeki (416,416) değerleri modelimizde kullanılan görsellerin boyutudur.
# 4. parametre ile BGR formattan RGB ye çeviririz. Son parametre de görselde bir kırpma yapmasını istemediğimizi belirtiyoruz.
# In [13]: img_blob.shape
# Out[13]: (1, 3, 416, 416)     böyle bir çıktı alırız. 4 boyutlu bir

    labels = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                        "trafficlight", "firehydrant", "stopsign", "parkingmeter", "bench", "bird", "cat",
                        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sportsball",
                        "kite", "baseballbat", "baseballglove", "skateboard", "surfboard", "tennisracket",
                        "bottle", "wineglass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                        "sandwich", "orange", "broccoli", "carrot", "hotdog", "pizza", "donut", "cake", "chair",
                        "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
                        "remote", "keyboard", "cellphone", "microwave", "oven", "toaster", "sink", "refrigerator",
                        "book", "clock", "vase", "scissors", "teddybear", "hairdrier", "toothbrush"]

    colors = ["0,255,255", "0,255,0", "255,0,0", "0,0,255", "255,255,0"]
    # üstteki değerler şu an string. İntegere'a çevirmemiz lazım.
    colors = [np.array(color.split(",")).astype("int") for color in colors]
    # her virgüle kadar olan değerleri böler ve integer'a çevirir. Üst satır. Bu bir kalıp ya lazım olur. 
    colors = np.array(colors)
    # üsttekinin açıklamasını tam yazmayacağım ama amaç değerleri tek array içine yazdırmak. Bölmüştük çünkü..
    colors = np.tile(colors, (18, 1))
    # np.tile ile colors sayılarını alt alta 18 kez array içinde büyültük, sağa doğru da 1 kez büyülttük.




    model = cv2.dnn.readNetFromDarknet("D:/YOLO/pretrained_model/yolov3.cfg", "D:/YOLO/pretrained_model/yolov3.weights")
# Modelimizi çalışmamıza ekliyoruz. ve ismini model yapıyoruz. !? :D bence güzel açıklama.

    model.setInput(frame_blob)
# şimdi blob tensör'ümüzü (yani görselimizi) modelimize verdik.

    layers = model.getLayerNames()
    # detection işlemi yapabilmek için modelimizdeki "layer"ları çekmemiz gerekiyor.
    # Tüm katmanlarımız burada. Buradaki bazı katmanlar bizim detection işlemimizin output katmanları.
    # Yani çıktılarımızın olduğu katmanlar da burada. Biz çıktı katmanlarını kullanacağız.Üst satır.
    
    #Öylesine bilgi: Tuple değiştirilemez, list değiştirilebilir.
    
    # output_layer = [layers[layer[0]-1] for layer in model.getUnconnectedOutLayers()]
    # Üstteki methodu ".getUnconnectedOutLayers()", terminal'e yazdığında "model"deki kaçıncı katmanların çıktı katmanı olduğunu söylüyor.
    # Çıktı olarak array([200, 227, 254]) yazıyor ancak ilk elemanı 1 aldığından dolayı. Bir önceki/bir eksik
    # layer ımız bizim çıktı katmanlarımız. Bir eksiğini alıyoruz. [0] ise aslında tüm değerler demek (!?)
    # sonuç olarak output layer larımızı output_layer'ımızda saklıyoruz. Çıktı katmanlarımızı tespit ettik.
    
    output_layer = [layers[199], layers[226], layers[253] ] #üstteki kısım yerine bunu yazdım.



    detection_layers = model.forward(output_layer) # yani aslında burada işte tüm ayarlamalardan sonra modelimize/cfg,weight'e resmi işletiyoruz.
    # Layer'ların yani işlem katmanlarını tutan değişkenimizi, hem modelimizi (cfg,weights) hem de giriş resmimizi verdiğimiz 
    #"model" değişkenimize uyguluyoruz ve "detection_layers" imizde sonuçları tutuyoruz. Bu sonuçlar Bounding Box' ve "Güven skorlarıdır"...

    ######## Non-Maximum-Suppression - Operation 1 ########
    
    ids_list = [] #predicted_id bilgilerini tutacak.
    boxes_list = [] # bounding_box bilgilerini tutacak.
    confidences_list = []# confidence bilgilerini tutacak.
    
    ############## END Of Operation - 1  ###############


    for detection_layer in detection_layers:
        for object_detection in detection_layer: #burayı değişkenlerin içine bakarak anlayabilirsin.
        
            scores = object_detection[5:] #ilk beş eleman bounding box'lar ile ilgili. Şu an amacımız güven skoru.
            predicted_id = np.argmax(scores) #.argmax bize en büyük elemanın index'ini verir.
            confidence = scores[predicted_id] # bulduğumuz index'e sahip elemanı tutuyoruz.
            
            if confidence > 0.05: # %30 güven skorundan yüksek orana sahip olanlar...
                
                label = labels[predicted_id] # Labels zaten algılanan cisimlerdi
                bounding_box = object_detection[0:4] * np.array([frame_width,frame_height,frame_width,frame_height])#Buradaki çarpım işlemi tamamen YOLO'nun arka plandaki matematiksel işlemi ile alakalı
                (box_center_x, box_center_y, box_width, box_height) = bounding_box.astype("int") #anlaşılabilir...
    #------------------------------------------------------------    
                start_x = int(box_center_x - (box_width/2)) #çizilecek olan box(dikdörtgenin) sol alt ve sağ üst köşelerinin koordinatları hesaplanıyor..
                start_y = int(box_center_y - (box_height/2))
    
    
                ######## Non-Maximum-Suppression - Operation 2 ######## burada oluşturduğumuz listelerin içini dolduruyoruz.
                
                ids_list.append(predicted_id)
                confidences_list.append(float(confidence)) #confidence değerleri çok değişik olabiliyor o yüzden float'a çevirdik.
                boxes_list.append([start_x, start_y, int(box_width), int(box_height)]) #append sonuna ekliyor. int'e çevirdik çünkü çizim yaparken float kullanılamıyor... e yani piksel koordinatları küsürlü olur mu krdşm... pls yani
    
                ################ END Of - Operation 2 ################


     

    ######## Non-Maximum-Suppression - Operation 3 ######## 
    
    max_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0,4) #Maks confidence'ye sahip olan bounding box'ları bize array liste şeklinde döndürür. 0.5 ve 0.4 değiştirilebilir ama uygulamaya göre değişir ama genel kullanım budur.
    
    for max_id in max_ids: 
        max_class_id = max_id #index'ten ziyade yani o değeri çekiyoruz. Kullanım böyle.
        box = boxes_list[max_class_id] # "bounding box"larımız içerisinden nesneye ait olan box'ları çektik.
        
        
        start_x = box[0]
        start_y = box[1]
        box_width = box[2]
        box_height = box[3]
    
        predicted_id = ids_list[max_class_id]
        label = labels[predicted_id]
        confidence = confidences_list[max_class_id]
    
    ################ END Of - Operation 3 ################ aşağıdakileri de for döngüsü içine aldık.
    
    
        end_x = start_x + box_width
        end_y = start_y + box_height
                
    #------------------------------------------------------------
        box_color = colors[predicted_id]   #
        box_color = [int(each) for each in box_color]
                
        label = "{}: {:.2f}%".format(label, confidence*100)
        print("predicted object {}".format(label))
                
        cv2.rectangle(frame, (start_x,start_y), (end_x,end_y), box_color, 1)
        cv2.putText(frame,label, (start_x, start_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
    
    
    #cv2.namedWindow('Detected', cv2.WND_PROP_FULLSCREEN) #Tam ekran yapılabilmesi için bir fonksiyon. Alttaki ile beraber kullanıldığı zaman.
    #cv2.setWindowProperty('Detected', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    
    cv2.imshow("Detected", frame)
    
   
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
#cv2.resizeWindow("Detected", 800, 600)


###    frame_width = frame.shape[1]

###    AttributeError: 'NoneType' object has no attribute 'shape'

# gibi hata alırsan bunu sebebi video okumasında sorun olduğundandır. Video uzantısını değiştir. Burada videoyu bir üst
# klasöre taşıyıp uzantıyı da ona göre düzenledik.



            