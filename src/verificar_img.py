# VERIFICAR SE MINHAS IMAGENS SÃO RGB OU TONS DE CINZA

import os
import cv2

def verificar_tipo_imagem(caminho_imagem):
    try:
        imagem = cv2.imread(caminho_imagem)

        if imagem is not None:
            altura, largura, canais = imagem.shape

            if canais == 1:
                print(f'A imagem {caminho_imagem} é em escala de cinza.')
            elif canais == 3:
                print(f'A imagem {caminho_imagem} é em cores (RGB).')
            else:
                print(f'A imagem {caminho_imagem} tem um número de canais não suportado.')
        else:
            print(f'A imagem {caminho_imagem} não pôde ser carregada.')
    except Exception as e:
        print(f'Erro ao processar a imagem {caminho_imagem}: {str(e)}')

diretorio_imagens = '/home/iza/Área de Trabalho/DisciplicaICA/n2_ica/imag'

conteudo_diretorio_principal = os.listdir(diretorio_imagens)

for item in conteudo_diretorio_principal:
    caminho_item = os.path.join(diretorio_imagens, item)

    if os.path.isdir(caminho_item):
     
        arquivos_no_diretorio = os.listdir(caminho_item)

        for nome_arquivo in arquivos_no_diretorio:
            caminho_imagem = os.path.join(caminho_item, nome_arquivo)

            if os.path.isfile(caminho_imagem) and nome_arquivo.lower().endswith(('.png', '.jpg', '.jpeg')):
                verificar_tipo_imagem(caminho_imagem)
            else:
                print(f'{caminho_imagem} não é um arquivo de imagem.')
