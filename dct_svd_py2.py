# -*- coding: utf-8 -*-
#!/usr/bin/env python3



# Trabalho da disciplina de álgebra linear computacional do Programa de Pós-Graduação em Modelagem Computacional da Universidade Federal de Juiz de Fora
# Alunos: Gisele Goulart e Ruan Medina


# Pacotes utilizados
import io
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
import math
#import urllib2
import IPython
from sklearn.metrics import mean_squared_error
from skimage import data
from skimage.color import rgb2gray
from numpy.linalg import svd
from skimage import img_as_ubyte,img_as_float
import glob

#==============================================================================

def load_images():
    list_of_images = glob.glob('./benchmarks/gray8bit/*.png')
    return list_of_images


# Abertura da imagem da URL passada como parametro
def get_image(list_images, k):
    #file_descriptor = urllib2.urlopen(image_url)
    #image_file = io.BytesIO(file_descriptor.read())
    image = Image.open(k)
    #img_color = image.resize(size, 1)
    img_grey = image.convert('L')
    img = np.array(img_grey, dtype=np.float)
    return img

# Aplicacao da tranformada DCT 2D 
def get_2D_dct(img):
    return fftpack.dct(fftpack.dct(img.T, norm='ortho').T, norm='ortho')

# Aplicacao da tranformada inversa DCT 2D
def get_2d_idct(coefficients):
    return fftpack.idct(fftpack.idct(coefficients.T, norm='ortho').T, norm='ortho')

# Reconstrucao da imagem
def get_reconstructed_image(raw):
    img = raw.clip(0, 255)
    img = img.astype('uint8')
    img = Image.fromarray(img)
    return img

# Decomposicao svd da imagem e truncamento com k valores singulares
def compress_svd(image,k):
    U,s,V = svd(image,full_matrices=False)
    reconst_matrix = np.dot(U[:,:k],np.dot(np.diag(s[:k]),V[:k,:]))
    rmse = math.sqrt(((pixels - reconst_matrix) ** 2).mean(axis=None))
    soma = np.sum(np.diag(s[:k]))
    return reconst_matrix, rmse, soma
    
# Compressao da imagem svd
def compress_show_gray_images(k, pixels):
    rmse_svd = []
    image=pixels
    original_shape = image.shape
    reconst_img, rmse, soma = compress_svd(image,k)
    compression_ratio =(original_shape[0]*original_shape[1])/(k*(original_shape[0] + original_shape[1]+1)) 
    return compression_ratio, rmse, reconst_img, soma

#==============================================================================    

#     
colors = ['b', 'r', 'g', 'y']
list_images = load_images()
# Loop no vetor de imagens
# Aplica SVD e DCT, mostra os resultados para cada imagem
for j in list_images:
    pixels = get_image(list_images, j)
    
    # Calculo das compressões (busca por erro fixado SVD)
    for i in range(1,257): 
        reconstructed_images = []
        rmse_dct = []
        compress_r_svd = []   
        compress_r_dct = [] 
        rmse_svd = []
        soma_svd=[]
        k_svd=[]
        k_dct=[]
        
        #Calculo Fatoração SVD
        compression_ratio_svd, rmse2, image_svd, soma = compress_show_gray_images(i, pixels)
        soma_svd.append(soma)
        
        # Análise de erro máximo na compressão
        if(rmse2<=20):
            k_svd.append(i)
            rmse_svd.append(rmse2)
            compress_r_svd.append(compression_ratio_svd)
            
#        if(rmse<=20):   
#            k_dct.append(i)
#            rmse_dct.append(rmse)    
#            compression_ratio_dct =(dct_size**2)/(i*i)
#            compress_r_dct.append(compression_ratio_dct)
        
        # Plot das imagens comprimidas
#        if((i%50)==0):
#            plt.close()
#            plt.title('DCT - r= '+str(i))
#            plt.imshow(reconstructed_images[i-1], cmap=plt.cm.gray)
#            plt.grid(False);
#            plt.xticks([]);
#            plt.yticks([]);
#            plt.show()
#            plt.title('SVD - r='+str(i))
#            plt.imshow(image_svd, cmap=plt.cm.gray)
#            plt.grid(False);
#            plt.xticks([]);
#            plt.yticks([]);
#            plt.show()        
        
        
    dct_size = pixels.shape[0]
    dct = get_2D_dct(pixels)
    #Calculo Transformada DCT
    dct_copy = dct.copy()
    dct_copy[i-1:,:] = 0
    dct_copy[:,i-1:] = 0 
    r_img = get_2d_idct(dct_copy);
    reconstructed_image_dct = get_reconstructed_image(r_img);
    rmse = math.sqrt(((pixels - r_img) ** 2).mean(axis=None))
    
    reconstructed_images.append(reconstructed_image_dct);                  # Criacao da lista de imagens
        

            
    print('**'+image_url[j][0]+'**')
    # Plot CMAP DCT
    print('**Distribuicao das frequencias da DCT**')
    plt.matshow(np.abs(dct[:50,:50]), cmap=plt.cm.Paired)
    plt.show()
    
    # Plot soma dos valores singulares
    plt.xlim(0,256)
    plt.title(str(image_url[j][0]))
    plt.plot(soma_svd, colors[j])
    plt.show()
    
    # Plot taxa de compressao X RMSE
    print('**Fixando RMSE em 20**')
    plt.figure()
    plt.grid()
    plt.plot(compress_r_svd, rmse_svd, '-o')
    plt.title('SVD - Posicoes em Memoria='+str(np.min(k_svd)*(256+256+1)))
    plt.xlabel("Taxa de Compressao")
    plt.ylabel("RMSE")
    plt.show()
    plt.figure()
    plt.grid()
    plt.plot(compress_r_dct, rmse_dct, '-o')
    plt.title('DCT - Posicoes em Memoria='+str(np.min(k_dct)**2))
    plt.xlabel("Taxa de Compressao")
    plt.ylabel("RMSE")
    plt.show()

#leg = ['Barco', 'Tabuleiro', 'Ressonancia', 'Lena']
#plt.legend(leg, loc='best')
#plt.grid()
#plt.show()



