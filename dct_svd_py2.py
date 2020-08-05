# -*- coding: utf-8 -*-
#!/usr/bin/env python3



# Trabalho da disciplina de álgebra linear computacional do Programa de Pós-Graduação em Modelagem Computacional da Universidade Federal de Juiz de Fora
# Alunos: Gisele Goulart e Ruan Medina


# Pacotes utilizados
import io
import os
from PIL import Image
import numpy as np
import pandas as pd
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
import seaborn as sns
import os
#==============================================================================

def load_images(path):
    list_of_images = glob.glob(path+'*.png')
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
    
    return reconst_matrix, rmse
    
# Compressao da imagem svd
def compress_show_gray_images(k, pixels):
    image=pixels
    original_shape = image.shape
    reconst_img, rmse = compress_svd(image,k)
    compression_ratio =(original_shape[0]*original_shape[1])/(k*(original_shape[0] + original_shape[1]+1)) 
    return compression_ratio, rmse, reconst_img
    
def r_definition(image, size, percent):
    U,s,V = svd(image,full_matrices=False)
    soma_total = np.sum(np.diag(s))
    r_search = 0
    for k in range(1,size):
        soma_parcial = np.sum(np.diag(s[:k]))  
        if(soma_parcial>=(percent*soma_total)):
            r_search = k
            break
        else:
            continue
    return r_search

#==============================================================================    

#     
pastas = [['Escala de Cinza','./benchmarks/gray8bit/'],
          ['Diversas','./benchmarks/misc/']]
percentuais=[0.5, 0.6, 0.7, 0.8, 0.9]
cols=['percent', 'rmse', 'meth', 'r', 'CR']
colors = ['b', 'r', 'g', 'y']

# Loop no vetor de imagens
# Aplica SVD e DCT, mostra os resultados para cada imagem
resultados = []
for database, path in pastas:
    list_images = load_images(path)
    for j in list_images:
        for p in percentuais:
            pixels = get_image(list_images, j)
            img_size = pixels.shape[0]
            r_min = r_definition(pixels,img_size, p)
            # Fatoracao SVD
            compression_ratio_svd, rmse_svd, reconstructed_image_svd = compress_show_gray_images(r_min, pixels)
            
            dct = get_2D_dct(pixels)
            
            #Calculo Transformada DCT
            dct_copy = dct.copy()
            dct_copy[r_min-1:,:] = 0
            dct_copy[:,r_min-1:] = 0 
            rec_img_comp = get_2d_idct(dct_copy);
            reconstructed_image_dct = get_reconstructed_image(rec_img_comp);
            rmse_dct = math.sqrt(((pixels - rec_img_comp) ** 2).mean(axis=None))
            compression_ratio_dct =(img_size**2)/(r_min*r_min)
            
            resultados.append([p,rmse_svd, 'svd', r_min, compression_ratio_svd])
            resultados.append([p,rmse_dct, 'dct', r_min, compression_ratio_dct])
            
            plt.title('SVD - r='+str(r_min))
            plt.imshow(reconstructed_image_svd, cmap=plt.cm.gray)
            plt.grid(False);
            plt.xticks([]);
            plt.yticks([]);
            plt.show() 
            n = j.split('/')[-1]
            n = n.replace('.png','')
            plt.imsave(str(database)+'_'+str(n)+'_svd_r'+str(r_min)+'_p'+str(p)+'.png', reconstructed_image_svd)
            
            plt.title('DCT - r='+str(r_min))
            plt.imshow(reconstructed_image_dct, cmap=plt.cm.gray)
            plt.grid(False);
            plt.xticks([]);
            plt.yticks([]);
            plt.show() 
            plt.imsave(str(database)+'_'+str(n)+'_dct_r'+str(r_min)+'_p'+str(p)+'.png', reconstructed_image_dct)
            
            plt.clf()
    data_results = pd.DataFrame(data=resultados, columns=cols)
    # SALVAR CSVS!!!!!!
    sns.set(style="ticks", palette="pastel")
    sns.boxplot(x="percent", y="rmse",
            hue="meth", palette=["m", "g"],
            data=data_results)
    sns.despine(offset=10, trim=True)
    plt.savefig(str(database)+'_rmse')
    plt.clf()
    # Calculo das compressões (busca por erro fixado SVD)
#    for i in range(1,257): 
#        reconstructed_images = []
#        rmse_dct = []
#        compress_r_svd = []   
#        compress_r_dct = [] 
#        rmse_svd = []
#        soma_svd=[]
#        k_svd=[]
#        k_dct=[]
        
        
        # Análise de erro máximo na compressão
#        if(rmse2<=20):
#            k_svd.append(i)
#            rmse_svd.append(rmse2)
#            compress_r_svd.append(compression_ratio_svd)
            
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
        
        

            
    #print('**'+image_url[j][0]+'**')
    # Plot CMAP DCT
#    print('**Distribuicao das frequencias da DCT**')
#    plt.matshow(np.abs(dct[:50,:50]), cmap=plt.cm.Paired)
#    plt.show()
    
    # Plot soma dos valores singulares
#    plt.xlim(0,256)
#    plt.title(str(image_url[j][0]))
#    plt.plot(soma_svd, colors[j])
#    plt.show()
#    
    # Plot taxa de compressao X RMSE
#    print('**Fixando RMSE em 20**')
#    plt.figure()
#    plt.grid()
#    plt.plot(compress_r_svd, rmse_svd, '-o')
#    plt.title('SVD - Posicoes em Memoria='+str(np.min(k_svd)*(256+256+1)))
#    plt.xlabel("Taxa de Compressao")
#    plt.ylabel("RMSE")
#    plt.show()
#    plt.figure()
#    plt.grid()
#    plt.plot(compress_r_dct, rmse_dct, '-o')
#    plt.title('DCT - Posicoes em Memoria='+str(np.min(k_dct)**2))
#    plt.xlabel("Taxa de Compressao")
#    plt.ylabel("RMSE")
#    plt.show()

#leg = ['Barco', 'Tabuleiro', 'Ressonancia', 'Lena']
#plt.legend(leg, loc='best')
#plt.grid()
#plt.show()



