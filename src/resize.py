import os
import cv2

diretorio_base = '/home/iza/Área de Trabalho/n2_ica/imag'

nova_altura, nova_largura = (224, 224)

for classe in ['AVCH', 'normal']:
    diretorio_classe = os.path.join(diretorio_base, classe)

    for img_name in os.listdir(diretorio_classe):
        caminho_imagem = os.path.join(diretorio_classe, img_name)

        imagem = cv2.imread(caminho_imagem)

        imagem_redimensionada = cv2.resize(imagem, (nova_largura, nova_altura), interpolation=cv2.INTER_CUBIC)

        cv2.imwrite(caminho_imagem, imagem_redimensionada)

        print(f'Imagem {img_name} redimensionada com sucesso para {nova_largura}x{nova_altura} pixels.')
