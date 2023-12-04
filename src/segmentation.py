### imports

import glob
import numpy as np
import pydicom as pdcm
import cv2 
import matplotlib.pyplot as plt 
import os

### Functions 

def maior_comp(image, n_comp=1):

    connectivity = 8

    output2 = cv2.connectedComponentsWithStats(image, connectivity, cv2.CV_16SC1)
    labels2 = output2[1]
    stats2 = output2[2]

    brain1 = np.zeros(image.shape, image.dtype)
    brain2 = np.zeros(image.shape, image.dtype)
    try:
        largecomponent21 = 1 + stats2[1:, cv2.CC_STAT_AREA].argmax()
        stats2[largecomponent21, cv2.CC_STAT_AREA] = largecomponent21

        largecomponent22 = 1 + stats2[1:, cv2.CC_STAT_AREA].argmax()
        stats2[largecomponent21, cv2.CC_STAT_AREA] = largecomponent22

    except ValueError:
        return np.zeros(image.shape)

    brain1[labels2 == largecomponent21] = 255
    brain2[labels2 == largecomponent22] = 255

    if n_comp == 1:
        return brain1
    else:
        return brain1 + brain2

def brain(img):
    # cv.imshow(' 1  Input image', img)

    #################################
    # 1 - Normalizacao
    #################################
    
    img_norm = np.zeros_like(img)
    cv2.normalize(img, img_norm, 0, 255, cv2.NORM_MINMAX)
    img_norm = cv2.convertScaleAbs(img_norm)

    ret, img_bin = cv2.threshold(img_norm, 230, 255, cv2.THRESH_BINARY)
    img_bin = maior_comp(img_bin)


    # cv.imshow(' 2 - Image bin', img_bin)
    
    # #########################################################
    # connectivity = 4
    # output = cv.connectedComponentsWithStats(img_bin, 4, cv.CV_8U)
    # labels = output[1]  # AM: Rotulo das componentes
    # stats = output[2]  # AM: Estatistica das componentes
    # centroids = output[3]  # AM: Centroids das componentes
    # #########################################################
    
    # img_max_ = np.zeros(img_bin.shape, img_bin.dtype)
    # img_max_2 = np.zeros(img_bin.shape, img_bin.dtype)

    # largecomponent1 = 1 + stats[1:, cv.CC_STAT_AREA].argmax()
    # largecomponent2 = 1 + stats[1:, cv.CC_STAT_AREA].argmax()

    # stats[largecomponent1, cv.CC_STAT_AREA] = largecomponent1
    # stats[largecomponent2, cv.CC_STAT_AREA] = largecomponent2

    # img_max_[labels == largecomponent1] = 255
    # img_max_2[labels != largecomponent2] = 255 

    kernel = np.ones((3, 3), np.uint8)
    
    # img_bin = cv2.erode(img_bin, kernel, iterations=1)
    img_bin = cv2.dilate(img_bin, kernel, iterations = 2)
    img_bin = cv2.erode(img_bin, kernel, iterations= 2)

    img_bin = cv2.dilate(img_bin, kernel, iterations = 8)
    img_bin = cv2.erode(img_bin, kernel, iterations= 2)

    img_bin = cv2.dilate(img_bin, kernel, iterations = 5)
    img_bin = cv2.erode(img_bin, kernel, iterations= 2)

    # cv.imshow(' 3 - Apenas o osso do cranio', img_bin)

    #Diferença  
    dif = img - img_bin
    # cv.imshow(' 4 - Diferenca 1 e 3', dif)


    dif = dif * 255
    ret, thresh = cv2.threshold(dif, 0, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    thresh = cv2.erode(thresh, kernel, iterations=2)

    # cv.imshow(' 5 - Thresh diferença', thresh)


    #################################
    # Detecção da maior componente
    #################################
    img_norm_2 = np.zeros_like(img)   

    #################################
    # Normalizacao da diferença
    #################################
    cv2.normalize(thresh, img_norm_2, 0, 255, cv2.NORM_MINMAX)
    img_norm_2 = cv2.convertScaleAbs(img_norm_2)
    if np.mean(img_norm_2) > 0.0:
        #########################################################
        connectivity = 4
        output = cv2.connectedComponentsWithStats(img_norm_2, 4, cv2.CV_8U)
        labels = output[1]  # AM: Rotulo das componentes
        stats = output[2]  # AM: Estatistica das componentes
        centroids = output[3]  # AM: Centroids das componentes

        #########################################################

        
        img_max_ = np.zeros(img_norm_2.shape, img_norm_2.dtype)

        largecomponent1 = 1 + stats[1:, cv2.CC_STAT_AREA].argmax()

        stats[largecomponent1, cv2.CC_STAT_AREA] = largecomponent1

        img_max_[labels == largecomponent1] = 255
        img_max_[labels != largecomponent1] = 0

        # cv.imshow(' 6 - Maior componente do thresh diferenca', img_max_)

        dif = (dif / 255)   
        # cv.imshow('  - TESTE', dif)

        img_max_ = dif - img_max_   
        # cv.imshow('  - TESTE - 2', img_max_)

        #################################
        # Normalizacao da diferença
        #################################
        img_norm_3 = np.zeros_like(img_max_)

        cv2.normalize(img_max_, img_norm_3, 0, 255, cv2.NORM_MINMAX)
        img_norm_3 = cv2.convertScaleAbs(img_norm_3)


        #################################    
        # Clip Contrast
        #################################
        mean, std = cv2.meanStdDev(img_norm_3) #Retirei por ultimo
        img_norm_3[img_norm_3 < mean] = 0

        # cv.imshow(' TESTE 3 imgnorm_3', img_norm_3)

        # cv.imshow(' TESTE 4 imgnorm_3/255', (img_norm_3/255))

        new_image = (dif - (img_norm_3))
        new_image[new_image < 0] = 0

        # cv.imshow(' 7 - Output image', new_image)

        # cv2.waitKey()
        return new_image
    else: 
        return None
    
def normalizeImage(v):
    result = ((v - v.min()) / (v.max() - v.min()) * 255).astype(np.uint8)
    return result

def get_imgs_path(root_folder):
    dicom_paths =[]
    for root, folder, files in os.walk(root_folder):    # generates the file names in a directory tree
        for file in files:
            nomes = os.path.join(root, file)            # create the path
            if nomes[-3:] == 'dcm':         
                dicom_paths.append(nomes)
    return dicom_paths               # append only the .dcm files

def segment_imgs(root_folder, output_folder):

    # Recebe os paths das imagens de entrada
    input_file_path_array = get_imgs_path(root_folder)

    # Definindo listas que vão ser populadas
    dicom_img_pixel_data_array = []
    windowed_img_hu_array = []
    segmented_img_hu_array = []
    
    for file_path in input_file_path_array:
        img_dicom = pdcm.dcmread(file_path)
        # Obter os valores de pixel da imagem
        pixel_array = img_dicom.pixel_array
        # Salvando os vetores em um array de vetores
        dicom_img_pixel_data_array.append(pixel_array)

        # Definir os parâmetros de janelamento (window center e window width)
        window_center = img_dicom.WindowCenter
        window_width = img_dicom.WindowWidth

        # Tipando corretamente os janelamentos caso venham com o tipo MultiValue:

        if isinstance(window_width, pdcm.multival.MultiValue) or isinstance(window_center, pdcm.multival.MultiValue):
            window_center = window_center[1]
            window_width = window_width[1]

        # Aplicar janelamento aos valores de pixel
        windowed_image = pixel_array * img_dicom.RescaleSlope + img_dicom.RescaleIntercept # y = ax + b

        # Ajustar os valores de pixel de acordo com os parâmetros de janelamento
        windowed_image = windowed_image.clip(min=window_center - window_width/2, max=window_center + window_width/2)



        # Salvando os vetores de imagens janeladas em um array 
        windowed_img_hu_array.append(windowed_image)
        
        # Aplica a função de segmentação
        brain_image = brain(windowed_image)
        if brain_image is not None:

            # # Salvando os vetores de imagens segmentadas em um array
            segmented_img_hu_array.append(brain_image)
            # # Exibir a imagem original e a imagem janelada
            # plt.figure(figsize=(12, 8))

            # plt.subplot(1, 2, 1)
            # plt.imshow(windowed_image, cmap='gray')
            # plt.title('Imagem Janelada')

            # plt.subplot(1, 2, 2)
            # plt.imshow(brain_image, cmap='gray')
            # plt.title('Imagem Segmentada')

            # plt.show()
        
            # Salvando as imagens segmentadas
            # for path in file_path:
            filename = file_path.split('\\')[-1]
            cv2.imwrite(output_folder + '\\' + filename[:-3] + 'png', normalizeImage(brain_image))
        else: 
            continue
    return dicom_img_pixel_data_array, windowed_img_hu_array, segmented_img_hu_array

def main():
    root = r'raw_data\Hemorrágico'
    output_data_folder = r'segmented_data\AVCH'
    array_normal_original, array_normal_janelada, array_normal_segmentada = segment_imgs(root, output_data_folder)

if __name__ == '__main__':

    main()    
