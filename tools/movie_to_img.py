import sys
import cv2
import numpy as np
import time
def main():

    argvs = sys.argv
    argc = len(argvs)

    if argc != 2:
        print('Usage: python {} (filename)'.format(arvs[0]))
        exit

    input_filename = argvs[1]

    cap = cv2.VideoCapture(input_filename)
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

    # 動画書込準備
    output_path = '/Users/downloads/belt_mov/vae_mov/'
    img_count = 0  # 保存した候補画像数
    frame_count = 0  # 読み込んだフレーム画像数
    num_cut = 5
    start = time.time()

    while(1):
        ret, frame = cap.read()
        k = cv2.waitKey(30) & 0xff
        if k == 27 or not ret:
            break
        cv2.imshow('frame', frame)
        
        if frame_count % num_cut == 0:
            img_file_name = output_path + str(img_count) + ".jpg"
            cv2.imwrite(img_file_name, frame)
            print(img_count)
            if img_count >= 180:
                break
            img_count += 1
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
