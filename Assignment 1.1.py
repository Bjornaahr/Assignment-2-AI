#%%
import numpy as np
from PIL import Image

#%%
image = Image.open('erna.jpg')
data = image.getdata()
width, height = image.size

#%%
red = [d[0] for d in data]
green = [d[1] for d in data]
blue = [d[2] for d in data]

mean = np.mean(data, axis=0)

redMatrix = np.zeros((height, width))
greenMatrix = np.zeros((height, width))
blueMatrix = np.zeros((height, width))

#%%
for i in range(height):
    for j in range(width):
        redMatrix[i,j] = red[i * width + j]
        greenMatrix[i,j] = green[i * width + j]
        blueMatrix[i,j] = blue[i * width + j]

#%%
def getChannelPCA(channel):


    cov_mat  = channel.T - np.mean(channel, axis=1)

    eig_vals, eig_vecs = np.linalg.eigh(np.cov(cov_mat))

    p = np.size(eig_vecs, axis = 1)

    idx = np.argsort(eig_vals) #Sorting eigenvalues/vecotrs in acending order
    idx = idx[::-1]
    eig_vecs = eig_vecs[:,idx]#Sorting eigenvectors
    eig_vals = eig_vals[idx]#Sorting eigenvalues

    numpc = 500

    if numpc < p or numpc > 0:
        eig_vecs = eig_vecs[:,range(numpc)]


    score = np.dot(eig_vecs.T, cov_mat)
    recon = np.dot(eig_vecs, score) + np.mean(channel, axis=1)
    newChannel = np.uint8(np.absolute(recon)).T

    return newChannel

#%%
rgbArray = np.zeros((height,width,3), 'uint8')
rgbArray[..., 0] = getChannelPCA(redMatrix)
rgbArray[..., 1] = getChannelPCA(greenMatrix)
rgbArray[..., 2] = getChannelPCA(blueMatrix)
img = Image.fromarray(rgbArray ,'RGB')
img.save("NewImage.png")
img.show()
#%%
