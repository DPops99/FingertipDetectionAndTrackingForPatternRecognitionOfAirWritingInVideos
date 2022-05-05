import sys
import os
import torch
import cv2

def fix_label(file_path):
    file = open(file_path, "r+")
    updated_lines = []
    for line in file.readlines():
        print(line)
        if len(line) != 0:
            update_line = "0" + line[1:]
            updated_lines.append(update_line)
    if len(updated_lines) != 0:
        file.seek(0)
        file.writelines(updated_lines)
    file.close()

def update_labels():
    for label_path in sys.argv[1:]:
        for (root, dirs, files) in os.walk(label_path, topdown=True):
            for file in files:
                fix_label(root+"/"+file)

def transform(img, mask):
    width = 320
    height = 480
    dim = (width, height)
    # resize image and mask
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
    mask = cv2.resize(mask, dim, interpolation=cv2.INTER_AREA).transpose(2, 0, 1)

    # convert to tensor
    img = torch.from_numpy(img).float()
    mask = torch.from_numpy(mask).float()

    # normalize
    img = img/255.0
    mask = mask/255.0

    return img, mask

if __name__ == "__main__":
    update_labels()
