import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import time
import cv2
from pathlib import Path
from os import listdir
from os.path import isfile, join

# -------------------------------------------------------------------------------------------
def append_list(name_list, list):

    for name_file in name_list:
        im = cv2.imread(mypath + name_file)
        list.append(im)
        # cv2.imshow("image", im)
        # cv2.waitKey(0)
    return list
# -------------------------------------------------------------------------------------------
def get_files(path):
    names_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    return names_files
# -------------------------------------------------------------------------------------------

# mypath = "./data/non-vehicles/GTI/"
# no_vehicle_names = get_files(mypath)
# nv_list = []
# nv_list = append_list(no_vehicle_names, nv_list)
#
# mypath = "./data/non-vehicles/Extras/"
# no_vehicle_names = get_files(mypath)
# nv_list = append_list(no_vehicle_names, nv_list)
#
# print(no_vehicle_names)
#
# with open('./data/No_Vehicle.pickle', 'wb') as handle:
#     pickle.dump(nv_list, handle, protocol=pickle.HIGHEST_PROTOCOL)


mypath = "./data/vehicles/GTI_Far/"
vehicle_names = get_files(mypath)
v_list = []
v_list = append_list(vehicle_names, v_list)

mypath = "./data/vehicles/GTI_Left/"
vehicle_names = get_files(mypath)
v_list = append_list(vehicle_names, v_list)

mypath = "./data/vehicles/GTI_MiddleClose/"
vehicle_names = get_files(mypath)
v_list = append_list(vehicle_names, v_list)

mypath = "./data/vehicles/GTI_Right/"
vehicle_names = get_files(mypath)
v_list = append_list(vehicle_names, v_list)
print(vehicle_names)

mypath = "./data/vehicles/KITTI_extracted/"
vehicle_names = get_files(mypath)
v_list = append_list(vehicle_names, v_list)
print(vehicle_names)


with open('./data/Vehicle.pickle', 'wb') as handle:
    pickle.dump(v_list, handle, protocol=pickle.HIGHEST_PROTOCOL)