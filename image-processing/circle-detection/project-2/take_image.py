import cv2
import random

vidcap = cv2.VideoCapture("./havuz.mp4")

total_frame = 150

last_image_index = 150

video_parameters = {"total_frame": int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)),
                    "FPS": int(vidcap.get(cv2.CAP_PROP_FPS)),
                    "frame_width":int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "frame_height": int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))}

randomFrameNumber =  random.sample(range(60, video_parameters['total_frame']), total_frame)

width = 416
height = 416
dim = (width, height)

index_of_img = 0
while index_of_img < total_frame:
    name = './data/all/' + str(last_image_index + index_of_img + 1) + '.jpg'
    print('Creating...' + name)
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, randomFrameNumber[index_of_img])
    success, image = vidcap.read()
    if success:
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        cv2.imshow(name, resized)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()  # Pencereyi kapat
        if key == ord('q'):
            cv2.imwrite(name, resized)
            index_of_img += 1
        else:  # q dışında bir tuşa basıldığında
            randomFrameNumber[index_of_img] = random.randint(60, video_parameters['total_frame'])
    else:
        break

cv2.destroyAllWindows()
