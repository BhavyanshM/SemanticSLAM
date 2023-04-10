import numpy as np
import cv2
import h5py
import os

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

def plot_masks(masks, image):
    if len(masks) == 0:
        return
    sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
    polygons = []
    color = []
    for ann in sorted_masks:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        mask_image = np.dstack((img, m*0.35))

        # Resize the mask_image to match the size of the input image
        mask_image = cv2.resize(mask_image, (image.shape[1], image.shape[0]))

        # Draw the mask as alpha 0.5 on the image
        image = cv2.addWeighted(image, 1, mask_image, 0.5, 0)

def dataset_main():
    home = os.path.expanduser('~')
    path = home + '/.ihmc/logs/perception/'
    filename = 'KITTI_Dataset_00.hdf5'
    group = '/kitti/left/'
    weights_file = "./yolov8n-seg.pt"

    # sam = sam_model_registry["vit_b"](checkpoint="../Weights/sam_vit_b_01ec64.pth")
    sam = sam_model_registry["vit_l"](checkpoint="../Weights/sam_vit_l_0b3195.pth")
    # sam = sam_model_registry["vit_h"](checkpoint="../Weights/sam_vit_h_4b8939.pth")

    mask_generator = SamAutomaticMaskGenerator(sam)
    

    data = h5py.File(path + filename, 'r')


    for index in range(len(data[group].keys())):

        print(data[group + str(index)])

        buffer = data[group + str(index)][:].view('uint8')
        buffer_image = np.asarray(buffer, dtype=np.uint8)
        buffer_image = cv2.imdecode(buffer_image, cv2.IMREAD_GRAYSCALE)
        buffer_image = cv2.cvtColor(buffer_image, cv2.COLOR_GRAY2RGB)

        masks = mask_generator.generate(buffer_image)

        plot_masks(masks, buffer_image)

        cv2.imshow("Frame", buffer_image)
        code = cv2.waitKeyEx(30)

        if code == 1048689:
            data.close()
            break
    
    data.close()

if __name__ == "__main__":
    dataset_main()