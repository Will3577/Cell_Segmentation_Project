from PIL import Image, ImageFont, ImageDraw, ImageOps
import os
from tqdm import tqdm
from numpy import loadtxt


def mk_dirs(path):
  if not os.path.isdir(path):
      os.makedirs(path)


save_dir = './Task3/'
mk_dirs(save_dir)
for folder_name in range(1,5):
    avg_size_list = loadtxt("./Sequences_p/average_size/0"+str(folder_name)+".txt", comments="#", delimiter="\n", unpack=False).astype('int')
    total_num_list = loadtxt("./Sequences_p/cell_count/0"+str(folder_name)+".txt", comments="#", delimiter="\n", unpack=False).astype('int')
    total_mitosis_list = loadtxt("./Sequences_p/mitosis_boundary/0"+str(folder_name)+"_file.txt", comments="#", delimiter="\n", unpack=False).astype('int')
    avg_disp_list = loadtxt("./Sequences_p/displacement/0"+str(folder_name)+"_dis.txt", comments="#", delimiter="\n", unpack=False)
    assert len(avg_size_list) == 92
    assert len(total_num_list) == 92
    assert len(total_mitosis_list) == 92
    assert len(avg_disp_list) == 91
    img_folder = './Sequences_p/mitosis_boundary/0'+str(folder_name)+'/'
    save_folder = save_dir+'0'+str(folder_name)+'/'
    mk_dirs(save_folder)

    for idx in tqdm(range(92)):
        avg_size = avg_size_list[idx]
        if idx==0:
            avg_disp = 0
        else:
            avg_disp = avg_disp_list[idx-1]
        total_num = total_num_list[idx]
        total_mitosis = total_mitosis_list[idx]
        img = Image.open(img_folder+"t"+"{0:0=3d}".format(idx)+".png")
        img = ImageOps.expand(img, border=45, fill=(255,255,255))
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("/System/Library/Fonts/NewYorkItalic.ttf", 36)
        text = "n_cells: "+str(total_num)+"      n_mitosis: "+str(total_mitosis)+"      avg_size: "+str(avg_size)+"      avg_disp: "+str(round(avg_disp, 2))
        draw.text((150,0),text,(0,0,0),font=font,align="center")
        img.save(save_folder+"t"+"{0:0=3d}".format(idx)+'.jpg')
