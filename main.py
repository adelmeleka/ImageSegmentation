from scipy.spatial import distance_matrix
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
import scipy.io
from PIL import Image
import math

def readImagesPath(path):
    dataName = path
    dataDir = "" + dataName
    trainData = []
    for ImageName in os.listdir(dataDir):
        ImagePath = os.path.join(dataDir, ImageName)
        trainData += [ImagePath]
    # return all images paths
    return trainData
# **************************************************************************************************
def imgRGBread(images):
    rgbImages = []  # 3D images
    vectorizedImages = []
    for i in range(0, len(images)):
        image = mpimg.imread(images[i])
        rgbImages.append(image)
        # Blur image to reduce the edge content and makes the transition form one color to the other very smooth.
        # Check video:https://www.youtube.com/watch?list=PLQVvvaa0QuDdttJXlLtAJxJetJcqmqlQq&v=sARklx6sgDk
        image = cv2.GaussianBlur(rgbImages[i], (7, 7), 0)

        # convert image to (M*N)*3 Matrix
        vectorizedImages.append(image.reshape(-1, 3))
    return rgbImages, vectorizedImages
# **************************************************************************************************
def imgRGBreadOneImage(imagePath):
    rgbImages = []  # 3D images
    vectorizedImage = []
    image = mpimg.imread(imagePath)
    rgbImages.append(image)
    # Blur image to reduce the edge content and makes the transition form one color to the other very smooth.
    # Check video:https://www.youtube.com/watch?list=PLQVvvaa0QuDdttJXlLtAJxJetJcqmqlQq&v=sARklx6sgDk
    image = cv2.GaussianBlur(image, (7, 7), 0)

    # convert image to (M*N)*3 Matrix
    vectorizedImage.append(image.reshape(-1, 3))
    return rgbImages, vectorizedImage
# **************************************************************************************************
def kmeans(dataSet, k):
    # I/p one of the images of vectorized Images list
    numOfPoints = len(dataSet)
    # k random initial points
    randomIndeces = np.random.choice(numOfPoints, k, replace=False)
    centers = []
    for i in range(0, len(randomIndeces)):
        centers.append(dataSet[randomIndeces[i]])
    centersOld = [0] * k
    clusterAssignment = [0] * len(dataSet)
    start = 0
    while (1):
        flag = 0
        if start != 0:
            for i in range(0, k):
                if centersOld[i] != centers[i]:
                    centersOld[i] = centers[i]
                else:
                    flag += 1
        start = 1
        if flag == k:
            return (centers, clusterAssignment)
            # distance between points and centers matrix
        distMatrix = distance_matrix(dataSet, centers, p=2)

        for i in range(0, numOfPoints):
            # closest center
            d = distMatrix[i]
            closestCenter = (np.where(d == np.min(d)))[0][0]
            # associate point to closest center
            clusterAssignment[i] = closestCenter

        # new centers
        for i in range(0, k):
            sumX = 0
            sumY = 0
            sumZ = 0
            count = 0
            for j in range(0, numOfPoints):
                if (clusterAssignment[j] == i):
                    sumX += (dataSet[j])[0]
                    sumY += (dataSet[j])[1]
                    sumZ += (dataSet[j])[2]
                    count += 1
            centers[i] = (sumX / count, sumY / count, sumZ / count)
    return (centers, clusterAssignment)
# **************************************************************************************************
def __extractGrondTruthMatrix(mat):
    _groundTruthMatrix = mat["groundTruth"][0][0][0][0][0]
    _groundTruthLabelVector = []
    for row in _groundTruthMatrix:
        _groundTruthLabelVector.extend(row.tolist())
    return _groundTruthMatrix,_groundTruthLabelVector
# **************************************************************************************************
def __getGroundTruthLabels(groundTruthMatrix,image):
    _labelsDict = {}
    i = -1
    for row in groundTruthMatrix:
        i += 1
        j = -1
        for key in row:
            j += 1
            if key not in _labelsDict:
                ima = image[i][j]
                _labelsDict.update({key:[ima[2],ima[1],ima[0]]})
    return _labelsDict
# **************************************************************************************************
def getGroundTruthLabelsAndGenerateImage(matPath,imagePath):

    image = cv2.imread(imagePath)
    mat = scipy.io.loadmat(matPath)
    groundTruthMatrix,_groundTruthLabelVector = __extractGrondTruthMatrix(mat)
    labelsDict = __getGroundTruthLabels(groundTruthMatrix,image)
    rowsNumber  = len(groundTruthMatrix)
    colsNumber = len(groundTruthMatrix[0])
    rgbArray = np.zeros((rowsNumber, colsNumber, 3), 'uint8')
    for i in range(rowsNumber):
        for j in range(colsNumber):
            rgbArray[i][j] = labelsDict[groundTruthMatrix[i][j]]
    img = Image.fromarray(rgbArray)
    img.save('groundTruth.jpg')

    return _groundTruthLabelVector
# **************************************************************************************************
def purityOfEachClass(labels,groundTruth2,k=3,sorted = True):
    groundTruthLabesNumber = 0
    for i in range(len(groundTruth2)):
        if groundTruthLabesNumber < groundTruth2[i]:
            groundTruthLabesNumber = groundTruth2[i]
    groundTruthLabesNumber +=1
    dataInClusterindexies = []
    for i in range(k):
        dataInClusterindexies.append([])
    for i in range(len(groundTruth2)):
        dataInClusterindexies[labels[i]].append(i)
    listNij = []
    for i in range(k):
        list = [0] * (groundTruthLabesNumber)
        listNij.append(list)

    for i in range(k):
        for j in range(len(dataInClusterindexies[i])):
            listNij[i][groundTruth2[dataInClusterindexies[i][j]]] += 1
    finalListNij = []
    for i in range(k):
        list = [0] * (groundTruthLabesNumber)
        finalListNij.append(list)

    for i in range(k):
        for j in range(groundTruthLabesNumber):
            finalListNij[i][j] = (listNij[i][j],j+1)

    if sorted == True:
        for i in range(k):
            finalListNij[i].sort(reverse=True)

    groundtruthList = [0] * (groundTruthLabesNumber)

    for i in range(k):
        for j in range(groundTruthLabesNumber):
            listNij[i][j] = finalListNij[i][j][0]

    for j in range(groundTruthLabesNumber):
        sum = 0
        for i in range(k):
            sum += finalListNij[i][j][0]
        groundtruthList[j] = sum

    return listNij,groundtruthList,groundTruthLabesNumber
#########################################################################################################################################################
def calculatePurity(labels,groundTruth,k=3):
    listNij,groundtruthList,groundTruthLabesNumber = purityOfEachClass(labels,groundTruth,k)
    sum = 0
    print(listNij)
    for i in range(k):
        sum += listNij[i][0]
    purity = sum / len(labels)
    return purity
#########################################################################################################################################################
def calculateF_Measure(labels,groundTruth,k=3):
    listNij, groundtruthList,groundTruthLabesNumber = purityOfEachClass(labels,groundTruth, k)
    NumberOfElementsInEachCluster = [0]*k
    for i in range(k):
        for j in range(len(listNij[i])):
            NumberOfElementsInEachCluster[i] += listNij[i][j]
    listF_measure = [0]*k
    for i in range(k):
        listF_measure[i] = 2*listNij[i][0]/(NumberOfElementsInEachCluster[i]+groundtruthList[i])
    sum = 0
    for i in range(k):
        sum += listF_measure[i]
    f_Measure = sum/k
    return f_Measure
#########################################################################################################################################################
def calculateConditionalEntropy(labels,groundTruth,k=3):
    listNij, groundtruthList,groundTruthLabesNumber = purityOfEachClass(labels,groundTruth, k,sorted=False)
    sizeOfData = len(groundTruth)
    numberOfElementsInEachCluster = [0]*k
    entropyOfEachCluster = [0]*k
    for i in range(k):
        for j in range(groundTruthLabesNumber):
            numberOfElementsInEachCluster[i] += listNij[i][j]
    for i in range(k):
        for j in range(groundTruthLabesNumber):
            if numberOfElementsInEachCluster[i] != 0:
                tempValue = listNij[i][j]/numberOfElementsInEachCluster[i]
            if tempValue != 0:
                entropyOfEachCluster[i] += (-tempValue)*math.log10((tempValue))

    entropy = 0
    for i in range(k):
        entropy += (numberOfElementsInEachCluster[i]/sizeOfData)*entropyOfEachCluster[i]
    return entropy

if __name__ == '__main__':
    trainImages = readImagesPath("data/images/train")
    matPath = "data/groundTruth/train/60079.mat"
    imagePath = "data/images/train/60079.jpg"
    groundTruthLabels = getGroundTruthLabelsAndGenerateImage (matPath,imagePath)
    rgbImages, vectorizedImages = imgRGBreadOneImage(imagePath)
    # print(vectorizedImages)

    # rgbImages, vectorizedImages = imgRGBread(trainImages)
    # print(len(groundTruthLabels))
    # # The vectorizedImages contain thee 200 train images each of 154401 vectors, each vector of 3 values,That's the I/p of the Kmeans
    #
    #
    k = 3
    testImage = vectorizedImages[0]
    finalCenters, clustersLabels = kmeans(testImage, k)

    print("ConditionalEntropy =  ",calculateConditionalEntropy(clustersLabels,groundTruthLabels))
    print("F_Measure =  ",calculateF_Measure(clustersLabels,groundTruthLabels))


    # # usage for sickit learn Kmeans
    # koko = KMeans(n_clusters=3).fit(testImage)
    # rec_img=koko.cluster_centers_[koko.labels_]
    # print(rec_img)
    # seg_im = rec_img.reshape((rgbImages[0].shape))
    # plt.imshow((seg_im/255))
    # plt.show()
    # plt.imshow(rgbImages[150])
    # plt.show()