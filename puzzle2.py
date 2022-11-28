import torch
from torchvision import transforms

import cv2
import pandas as pd
import numpy as np

from matplotlib.path import Path
import matplotlib.pyplot as plt
from PIL import Image

from basic import * 
import config
import pymysql 
import pandas as pd
from datetime import timedelta, date, datetime
import shutil

device = torch.device('cpu')
def connection():
    conn = pymysql.connect(host=config.host,
                            user=config.username,
                            passwd=config.password, 
                            db=config.database, 
                            port=config.port, charset='utf8')
    cursor = conn.cursor()
    return conn, cursor

class DetectCriminalSign:

    def __init__(self, sidewalk_df_path, sidewalk_dic_path,
                       stopline_df_path, stopline_dic_path,
                       light_df_path):

        self.sidewalk_dic_path = sidewalk_dic_path
        self.sidewalk_df = pd.read_csv(sidewalk_df_path) # 보행자도로 point df
        self.sidewalk_dict = load_pickle(sidewalk_dic_path) # 보행자도로 mask pickle

        self.stopline_dic_path = stopline_dic_path
        self.stopline_df = pd.read_csv(stopline_df_path) # 보행자도로 point df
        self.stopline_dict = load_pickle(stopline_dic_path) # 보행자도로 mask pickle

        self.light_df = pd.read_csv(light_df_path)

        # 신호등 판별 모델
        self.light_cf_model = torch.load('./light_classfication.pth', map_location=device)
        self.detect_motorcycle_model = torch.hub.load('ultralytics/yolov5', 'custom', path='detect_motorcycle.pt')
        
        self.conn, self.cursor = connection()

    # DB Insert 함수
    # input : 위반 정보
    # output : 업데이트
    def insert_data_to_DB(self, value):

        # ex ) ["보행자 도로 위반", 4, '2022-09-27 09:10:01', 'C000002']
        print(value)
        insert_sql = f"insert into crime(crime_type,crime_cnt,time,cctv_id) values ('{value[0]}', {value[1]},'{value[2]}','{value[3]}');"

        self.cursor.execute(insert_sql)
        self.conn.commit()
        print("INSERT_완료", value[0], value[1])

    # 비디오에서 사진 뽑아오는 함수
    # input : 비디오 경로, 저장할 경로
    # output : (이미지 자체 저장)
    def capture_img(self, video_path, save_path):
        
        video_name = video_path.split('_')[0]
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)

        count = 0 # 추출한 이미지 개수 
        while (video.isOpened()): # 영상이 열려있는 동안 계속 진행
            ret, image = video.read()
            if(int(video.get(1)) % fps == 0): #앞서 불러온 fps 값을 사용하여 1초마다 추출
                cv2.imwrite(save_path + f"/{video_name}_{count}.jpg", image)
                count += 1
            
            if not ret: # return 값이 없으면 동영상이 끝났다는 뜻 
                print(f"추출완료.\n총 {count}개 추출")
                break

        video.release()
    
    # mask를 만들어서 Dictionary에 저장 후 return 
    # input : mode (정지선인지, 보행자도로인지), video_code
    # output : (dictionary 업데이트) 
    def make_mask(self, mode: str, video_name:str, ORI_WIDTH = 1920, ORI_HEIGHT = 1080):
        # 비디오 이름으로 데이터 좌표 찾아오기 
        if mode == 'sidewalk':
            tmp_df = self.sidewalk_df[self.sidewalk_df['video_code'] == video_name]['points']
        elif mode == 'stopline':
            tmp_df = self.stopline_df[self.stopline_df['video_code'] == video_name]['points']

        # mask로 만들 기초 
        x, y = np.mgrid[:ORI_HEIGHT, :ORI_WIDTH]
        coors = np.hstack((x.reshape(-1, 1), y.reshape(-1,1))) # coors.shape is (4000000,2)

        # 보행자 도로 mask 만들기 
        for i, value in enumerate(tmp_df.values):

            sidewalk_points = value
            # 문자열처리
            sidewalk_points = sidewalk_points.replace("'", '').replace("[", '').replace("]", '')
            sidewalk_points = sidewalk_points.split(', ')
            sidewalk_points = np.array(list(map(int,map(float, sidewalk_points))))
            # [[Y, X], [Y, X]]-> [[X, Y], [X, Y]]
            sidewalk_points = sidewalk_points.reshape(-1, 2)
            sidewalk_points = np.array(list(zip(sidewalk_points[:, 1], sidewalk_points[:, 0])))

            if i == 0: # 이거는 보행자 도로가 여러 개 일때 초기식
                poly_path=Path(sidewalk_points)
                mask = poly_path.contains_points(coors)
            else: # 보행자 도로 여러개 일때 합치기 
                poly_path=Path(sidewalk_points)
                mask += poly_path.contains_points(coors)

        mask = mask.reshape(-1, ORI_WIDTH)
        if mode == 'sidewalk':
            self.sidewalk_dict[video_name] = mask
            save_pickle(self.sidewalk_dic_path, data = self.sidewalk_dict)

        elif mode == 'stopline':
            self.stopline_dict[video_name] = mask
            save_pickle(self.stopline_dic_path, data = self.stopline_dict)

    # mask 위치 show()  
    # input : mode (정지선인지, 보행자도로인지), video_code
    # output : (자체 출력)
    def show_mask(self, mode, video_code):
        if mode == 'sidewalk':
            if video_code in self.sidewalk_dict:
                plt.figure(figsize=(12, 8))
                plt.imshow(self.sidewalk_dict[video_code])
                plt.axis('off')
            else:
                print(f"{video_code} : 해당 video에 대한 보행자 도로에 정보가 없어 확인이 필요합니다. ")
        elif mode == 'stopline':
            if video_code in self.stopline_dict:
                plt.figure(figsize=(12, 8))
                plt.imshow(self.stopline_dict[video_code])
                plt.axis('off')
            else:
                print(f"{video_code} : 해당 video에 대한 정지선 정보가 없어 확인이 필요합니다. ")

    # 오토바이 모델 탐지 모델 
    # input : image
    # output : detect result (DataFrame)
    def detect_motorcycle(self, image):
        result = self.detect_motorcycle_model(image)
        # result.save(save_dir='results/res')
        result = result.pandas().xywh[0] 
        # 확률이 50%이상인 데이터만
        result = result[result['confidence'] >= 0.5]
        result = result[result.columns[:4]]
        result = result.astype(int)

        return result      
    
    # 탐지한 오토바이의 영역을 rect형태의 mask로 변환시켜주어야 함 
    # input : value ([X, Y, W, H]) 왼쪽 상단 점, 너비, 높이 
    # output : 오토바이_mask
    def make_motorcycle_mask(self, value, ORI_WIDTH = 1920, ORI_HEIGHT = 1080):
        x1, y1 = value[0], value[1]
        x2, y2 = x1 + value[2], y1 + value[3]

        moto_image = np.full((ORI_HEIGHT,ORI_WIDTH), 0, np.uint8)
        cv2.rectangle(moto_image, (x1, y1), (x2, y2), 1, -1)

        return moto_image

    # 오토바이가 SEG에 차지하는 영역이 어느 정도인지 판단하는 함수
    # input : mask(2차원 numpy 배열), motorcycle(오토바이의 위치가 저장된 2차원 numpy 배열)
    # output : 오토바이 역역에서의 겹치는 부분 비율 
    def percent_rect_area_mask(self, mask, motorcycle):

        all_area_moto = motorcycle.sum() # 오토바이 영역의 합
        sub = (mask & motorcycle).sum() # 겹치는 부분의 합   

        return sub / all_area_moto * 100 # 오토바이 역역에서의 겹치는 부분 비율 

    # 보행자도로 범법행위 탐지 모델
    # input : 비디오 경로
    # output : 몇 개 범법행위 했는지 Dictionary
    def sidewalk_criminal_detect(self, image_paths:list):

        video_name = image_paths[0].split('/')[-1].split('_')[0]

        # 도로에 보행자도로가 없는 경우 
        if video_name not in self.sidewalk_dict:
            print(f'{video_name} : 이 CCTV에서는 보행자도로가 없어 탐지가 불가능합니다.')
            return 0

        result = {'STATE_0' : 0, 'STATE_1':0, 'STATE_2':0}

        # 이미지의 목록을 모두 돌리면서 
        for image in image_paths:
            detect_result = self.detect_motorcycle(image) # 예상 output = [x, y, w, h]  / res.xywh
            print("*************",image,"*************")
            # print(detect_dresult)
            result = self.count_motorcycle(detect_result, video_name, 'sidewalk', result)
            # print(result)
            # print()
        time_value = datetime.now()
        value = ["보행자 도로 위반", result['STATE_2'], time_value, 30]
        self.insert_data_to_DB(value)
        
    '''                        여기까지 보행자 도로 위반에 필요한 함수들                            '''
    ##########################################################################################
    '''                        여기서 부터  정지선 위반 구역에 대한 코드                           '''

    # 신호등 판별을 위해 이미지에서 신호등 위치만 크롭
    # input : 이미지, 신호등 위치 list
    # output : 신호등 판별 결과
    def crop_light_image(self, img, value):
        wantedSize = 256
        x, y, w, h = int(value[1]), int(value[2]), int(value[3]), int(value[4])
        # 자르기
        image = img[y: (y + h), x: (x + w)]
        image = Image.fromarray(image)

        # 256 * 256 으로 바꾸기
        image = image.resize((wantedSize, wantedSize))

        result = self.clasffication_light(image) #신호등 판별
        
        return result 

    # 신호등 판별 
    # input : 비디오 경로
    # output : 몇 개 범법행위 했는지 Dictionary
    def clasffication_light(self, crop_img):
        # 이미지 전처리
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = preprocess(crop_img)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

        with torch.no_grad():
            output = self.light_cf_model(input_batch)
            clsidx = torch.argmax(output)  
            result = clsidx.item()

        return result

    # 데이터 프레임 돌면서 cnt 
    # input : 오토바이 탐지 결과 df, video_code, 정지선 or 보행자, 결과 딕셔너리
    # output : 결과 딕셔너리 
    def count_motorcycle(self, df, video_name, mode, result):

        if mode == 'stopline':
            mask = self.stopline_dict[video_name]
        elif mode == 'sidewalk':
            mask = self.sidewalk_dict[video_name]
        
        for res in df.values:
            motorcycle_loc = (res[0], res[1], res[2], res[3])
            motorcycle_mask = self.make_motorcycle_mask(motorcycle_loc)

            bottom_left = (motorcycle_loc[0], motorcycle_loc[1] + motorcycle_loc[3]) # (x, y + h)
            bottom_right = (motorcycle_loc[0] + motorcycle_loc[2], motorcycle_loc[1] + motorcycle_loc[3]) # (x + w, y + h)

            # STATE 2 - First Condition 
            # Bottom 좌표가 mask에 포함되는 경우
            if (mask[bottom_left[1]-1, bottom_left[0]-1] != 0) & (mask[bottom_right[1]-1, bottom_right[0]-1] != 0):
                print("State 2 _ condition 1 CHECK")
                result['STATE_2'] += 1

            # STATE 2 - Second Codition 
            # 오토바이와 Seg영역이 60%이상 겹칠 때 
            elif self.percent_rect_area_mask(mask, motorcycle_mask) >= 60:
                print("State 2 _ condition 2 CHECK", self.percent_rect_area_mask(mask, motorcycle_mask))
                result['STATE_2'] += 1
            
            # STATE 1 - First Codition 
            # 바닥 좌표중 1개와 겹치면서 겹치는 영역이 50% 이상일 때 
            elif ((mask[bottom_left[1]-1, bottom_left[0]-1] != 0) | (mask[bottom_right[1]-1, bottom_right[0]-1] != 0)) & \
                (self.percent_rect_area_mask(mask, motorcycle_mask) >= 40):
                result['STATE_1'] += 1
            
            # STATE 0 - 위반 아님 
            else:
                result['STATE_0'] += 1
            
            return result

    def stopline_criminal_detect(self, image_paths):

        video_name = image_paths[0].split('/')[-1].split('_')[0]
        
        # CCTV에 신호등이 없는 경우 
        if video_name not in self.light_df['video_code'].unique():
            print(f'{video_name} : 이 CCTV에서는 신호등 정보가 없어 탐지가 불가능합니다.')
            return 0
        
        light_loc = self.light_df[self.light_df['video_code'] == video_name].values[0]

        cnt = 0
        for image_path in image_paths:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            light_result = self.crop_light_image(image, light_loc)
            if light_result == 0:
                print("초록불")
                cnt = 0
                # shutil.rmtree('./results/') # 해당 폴더 삭제하기
            else: 
                print("빨간불")
                # cnt += 1
                # if cnt == 3:
                # <1> 오토바이 detect모델 돌려서
                detect_result = self.detect_motorcycle(image)
                # <2> 정지선 위반 구역에 있는지 확인하기 
                result = {'STATE_0' : 0, 'STATE_1':0, 'STATE_2':0}
                result = self.count_motorcycle(detect_result, video_name, 'stopline', result)

                time_value = datetime.now()
                value = ["정지선 위반", result['STATE_2'], time_value, 30]
                self.insert_data_to_DB(value)
                

