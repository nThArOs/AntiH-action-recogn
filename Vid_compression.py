from coviar import load
from coviar import get_num_gops
from coviar import get_num_frames
import os
import subprocess
from PIL import Image
import numpy as np
import cv2

data_path = "antihpert-gre/test"
output_path = os.path.join("compressed",data_path)
gop_size = 10

for folder in os.listdir(data_path):
    folder_path = os.path.join(output_path, folder)
    origin_path = os.path.join(data_path, folder)
    if not os.path.exists(os.path.join(output_path,folder)):
        os.makedirs(folder_path)
    for vid in os.listdir(origin_path):
        vid_path = os.path.join(origin_path, vid)
        raw_vid = "temp" + "_" + vid
        raw_vid_path = os.path.join(folder_path,raw_vid)
        compressed_vid = "compressed" + "_" + vid
        compressed_vid_path = os.path.join(folder_path, compressed_vid)
        if not os.path.exists(raw_vid_path):
            cmd_vid_to_raw = f"ffmpeg -i {vid_path} -vf scale=340:256 -q:v 1 -c:v mpeg4 -f rawvideo {raw_vid_path}"
            subprocess.run(cmd_vid_to_raw, shell=True, check=True)
        num_gops = get_num_gops(raw_vid_path)
        num_frames = get_num_frames(raw_vid_path)
        gop_index = 0
        frame_index = 0
        imgid = 0
        while (gop_index < num_gops):
            if gop_index + 1 == num_gops:
                overly_frame = (num_gops * (gop_size + 2)) - num_frames
                while (frame_index <= ((gop_size + 2) - overly_frame) - 1):
                    print(f"Gop {frame_index} / {gop_index} / {imgid:06d}")
                    if frame_index == 0:
                        raw_iframe = load(raw_vid_path, gop_index, frame_index, 0, True)
                        raw_iframe = cv2.cvtColor(raw_iframe, cv2.COLOR_BGR2RGB)
                        img_iframe = Image.fromarray(raw_iframe)
                        image_iframe = img_iframe.save(f"{folder_path}/{imgid}.jpg")
                    else:
                        raw_residual = load(raw_vid_path, gop_index, frame_index, 2, True)
                        img_residual = Image.fromarray((raw_residual * 255).astype(np.uint8))
                        image_residual = img_residual.save(f"{folder_path}/{imgid}.jpg")
                    frame_index += 1
                    imgid += 1
                gop_index += 1
            else:
                while (frame_index <= gop_size + 1):
                    print(f"1111Gop {frame_index} / {gop_index} / {imgid:06d}")
                    if frame_index == 0:
                        raw_iframe = load(raw_vid_path, gop_index, frame_index, 0, True)
                        raw_iframe = cv2.cvtColor(raw_iframe, cv2.COLOR_BGR2RGB)
                        img_iframe = Image.fromarray(raw_iframe)
                        image_iframe = img_iframe.save(f"{folder_path}/{imgid}.jpg")
                    else:
                        raw_residual = load(raw_vid_path, gop_index, frame_index, 2, True)
                        img_residual = Image.fromarray((raw_residual * 255).astype(np.uint8))
                        image_residual = img_residual.save(f"{folder_path}/{imgid}.jpg")
                    frame_index += 1
                    imgid += 1
                if frame_index > gop_size:
                    frame_index = 0
                    gop_index += 1
        print("make vid")
        os.remove(raw_vid_path)
        if not os.path.exists(compressed_vid_path):
            cmd_create_vid_from_img = f"ffmpeg -framerate 30 -pattern_type glob -i '{folder_path}/*.jpg' {compressed_vid_path}"
            subprocess.run(cmd_create_vid_from_img, shell=True, check=True)

        print(folder_path)
        for item in os.listdir(folder_path):
            if item.endswith(".jpg"):
                os.remove(os.path.join(folder_path,item))