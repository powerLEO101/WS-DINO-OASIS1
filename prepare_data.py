from os.path import join, exists
from os import listdir, mkdir
import numpy as np
import nibabel as nb
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

df = pd.read_csv('../Data/oasis_cross-sectional.csv')

def to_gray(im):
    im_np = (im * 255) / im.max()
    im_np = im_np.astype(np.uint8)
    return im_np
def extract_disc(disc_name, output_dir, input_dir):
    path = join(input_dir, disc_name)
    for f1 in listdir(path):
        tmp_name = f1 + '_mpr_n4_anon_111_t88_masked_gfc.img'
        f = join(path, f1, 'PROCESSED/MPRAGE/T88_111/', tmp_name)
        if exists(f) == False:
            tmp_name = f1 + '_mpr_n3_anon_111_t88_masked_gfc.img'
            f = join(path, f1, 'PROCESSED/MPRAGE/T88_111/', tmp_name)
        if exists(f) == False:
            tmp_name = f1 + '_mpr_n6_anon_111_t88_masked_gfc.img'
            f = join(path, f1, 'PROCESSED/MPRAGE/T88_111/', tmp_name)
        if exists(f) == False:
            tmp_name = f1 + '_mpr_n5_anon_111_t88_masked_gfc.img'
            f = join(path, f1, 'PROCESSED/MPRAGE/T88_111/', tmp_name)

        tmp_img = nb.load(f)
        tmp_data = tmp_img.get_fdata()
        if exists(output_dir) == False:
            print('dir does not exist')
            exit(1)
        im_x = []
        for index in range(73, 103):
            slice = tmp_data[index, 16:192, :].reshape(176, 176)
            slice = to_gray(slice)
            im_x.append(slice)
        im_y = []
        for index in range(89, 119):
            slice = tmp_data[:, index, :].reshape(176, 176)
            slice = to_gray(slice)
            im_y.append(slice)
        im_z = []
        for index in range(73, 103):
            slice = tmp_data[:, 16:192, index].reshape(176, 176)
            slice = to_gray(slice)
            im_z.append(slice)
        im_x = np.array(im_x, dtype=np.uint8)
        im_y = np.array(im_y, dtype=np.uint8)
        im_z = np.array(im_z, dtype=np.uint8)

        np.save(join(output_dir, f1 + '_x'), im_x)
        np.save(join(output_dir, f1 + '_y'), im_y)
        np.save(join(output_dir, f1 + '_z'), im_z)


for i in range(1, 13):
    extract_disc('disc' + str(i), '/home/leo101/Work/Harp/Data/OASIS_NEW', '/home/leo101/Work/Harp/Data1')
    print('disc' + str(i) + ' extracted')
