import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 由于人脸重建的时候涉及到随机选取人脸
# 因此需要设定随机数种子，这里使用time返回的数字作为种子
import time
np.random.seed(int(time.time()))

# ORL数据集一共有400张图片，在这里可以分配训练集和测试集的大小
# trainImgsPerPerson用以表示数据集中每个人的10张照片中有多少张作为训练集
trainSetSize = 320
testSetSize = 80
trainImgsPerPerson = trainSetSize//40


def getFacesImgPath(ORLFolderPath):
    '''
    输入ORL数据集根目录
    输出数据集中所有人脸图像（bmp格式）的路径
    ORL数据集含有40个人，每人10张照片，共400张图像
    '''
    # 此行用以获取每个subject的图像子文件夹路径
    faceFolders = glob.glob(ORLFolderPath+'/*')
    facesImgPath = []
    # 该for循环用以获取每个subject对应子文件夹中的图像文件路径，并进行汇总
    for faceFolder in faceFolders:
        facesImgPath.extend(glob.glob(faceFolder+'/*.bmp'))
    assert len(facesImgPath) == 400
    return facesImgPath


def getTrainSetAndTestSet(facesImgPath):
    '''
    输入所有人脸图像的路径构成的list
    输出训练集和测试集的图像list
    '''
    trainSet, testSet = [], []
    trainLabel, testLabel = [], []
    for i in range(40):
        for j in range(10):
            # 在ubuntu系统下，下面这一行需要稍作修改，此行用以获取对应subject的编号，将其视作label
            label = int(facesImgPath[i*10+j].split('\\')[1][1:])
            # 前（10 - trainImgsPerPerson）作为测试集，后trainImgsPerPerson张作为训练集
            if j >= trainImgsPerPerson:
                testSet.append(facesImgPath[i*10+j])
                testLabel.append(label)
            else:
                trainSet.append(facesImgPath[i*10+j])
                trainLabel.append(label)
    assert len(trainSet) == trainSetSize
    assert len(testSet) == testSetSize
    return trainSet, testSet, trainLabel, testLabel

# 使用上面的两个函数获取训练集和测试集图像路径list及其对应的label
facesImgPath = getFacesImgPath('../ORL')
trainSet, testSet, trainLabel, testLabel = getTrainSetAndTestSet(facesImgPath)

# 根据路径读取图像文件，设灰度图像（若非灰度图，则先行转化为灰度图）的shape为(H,W)，将其flatten为H*W长度的向量
# trainData的每一列即为一个训练样本
trainData = np.zeros((112*92, trainSetSize))
for idx, faceFile in enumerate(trainSet):
    img = cv2.imread(faceFile)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    trainData[:, idx] = img.flatten()
# testData的每一列即为一个测试样本，自然，也需要先行转化为灰度图
testData = np.zeros((112*92, testSetSize))
for idx, faceFile in enumerate(testSet):
    img = cv2.imread(faceFile)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    testData[:, idx] = img.flatten()


# 获取、查看并保存平均脸
trainDataMean = np.mean(trainData, axis=1, keepdims=True)
trainDataMean = np.array(trainDataMean, dtype=np.uint8)
cv2.imshow('meanFace', np.resize(trainDataMean, (112, 92)))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('./meanFace.png', np.resize(trainDataMean, (112, 92)))


# 计算协方差矩阵，以及其特征值和特征向量，这里使用数学技巧进行转化，降低计算复杂度
trainDataMean = np.mean(trainData, axis=1, keepdims=True)
trainData = trainData - trainDataMean
trainDataCovarianceMatrix = np.matmul(np.transpose(trainData), trainData)/trainSetSize
eigenValues, eigenVectors = np.linalg.eig(trainDataCovarianceMatrix)
eigenVectors = np.matmul(trainData, eigenVectors)
# 归一化这一步很重要，因为计算得到的特征向量并非单位向量，但是我们需要一个标准的度量
eigenVectors = eigenVectors/np.linalg.norm(eigenVectors, axis=0, keepdims=True)


# 可视化特征脸
for i in range(20):
    eigenFace = eigenVectors[:, i]
    # 可视化之前需要对数据进行归一化处理，方可正常显示
    eigenFace = (eigenFace - np.min(eigenFace))/(np.max(eigenFace)-np.min(eigenFace))
    eigenFace = np.resize(eigenFace, (112, 92))

    # 使用pyplot能够得到热图化输出，更加好看
    plt.figure('eigenFace')
    plt.imshow(eigenFace)
    plt.axis('off')
    plt.show()
    plt.imsave('./eigenFace'+str(i).zfill(2)+'.png', eigenFace)


# 人脸重建
# 随机选取一张人脸图像，读取并灰度化，保存该原始图像
faceChooseToReconstruct = facesImgPath[np.random.randint(0, 360)]
faceChooseToReconstruct = cv2.imread(faceChooseToReconstruct)
faceChooseToReconstruct = cv2.cvtColor(faceChooseToReconstruct, cv2.COLOR_BGR2GRAY)
cv2.imwrite('./originFace.png', faceChooseToReconstruct)
# 将原始图像拉成一个长向量并减去均值
faceChooseToReconstruct = np.array(faceChooseToReconstruct.flatten(), dtype=np.float)-trainDataMean.flatten()
# projectDims为不同的投影维度数
projectDims = [10, 25, 40, 55, 70, 85, 100, 115, 130, 145, 160, 175, 190, 205, 220, 235, 250, 265, 280, 295, 310, 325, 340, 355]
for dim in projectDims:
    # 对于每一哥投影维度，将原始图像投影至对应维度的主空间中，得到投影值
    projectValues = np.matmul(np.transpose(faceChooseToReconstruct), eigenVectors[:,0:dim])
    reconstructFace = np.zeros(112*92)
    # 使用投影值，以及每个投影值对应的特征向量（即eigenface）进行人脸重建
    for i in range(dim):
        reconstructFace = reconstructFace + projectValues[i]*eigenVectors[:, i]
    # 可视化之前需要对数据进行归一化处理，方可正常显示
    reconstructFace = (reconstructFace - np.min(reconstructFace))/(np.max(reconstructFace)-np.min(reconstructFace))
    reconstructFace = np.resize(reconstructFace, (112, 92))

    # 显示重建的人脸并保存
    cv2.imshow('reconstructFace', reconstructFace)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    reconstructFace = np.array(np.clip(reconstructFace*255, 0 ,255), dtype=np.uint8)
    cv2.imwrite('reconstructFace'+str(dim).zfill(3)+'.png', reconstructFace)


# 人脸识别
# 这里是使用9:1设置训练集与测试集的比例得到的识别效果，主要是为了探究投影维度对识别效果的影响
# 我们手上有的仅仅是训练集上的数据，因此减去训练集的均值向量
testData = testData - trainDataMean
accuracy = []
for projectDim in range(1, trainSetSize+1):
    # 训练集中每个人脸都向特征向量进行投影，分别顺序投影至1维和360维特征空间，以便观察识别准确率与投影维度的关系
    # 将所有的训练数据投影至特征空间
    trainSetProjections = np.transpose(np.matmul(np.transpose(trainData), eigenVectors[:, :projectDim]))
    # 将所有的测试数据投影至特征空间
    testSetProjections = np.transpose(np.matmul(np.transpose(testData), eigenVectors[:, :projectDim]))
    predictRightCount = 0
    # 对每一条测试数据，判断其投影值和训练数据的投影值中的哪一条欧氏距离最接近
    for i in range(testSetSize):
        testSetProjection = testSetProjections[:, i]
        testSetProjection = testSetProjection[:, np.newaxis]
        compareResult = np.linalg.norm(trainSetProjections - testSetProjection, axis=0)
        predictRightCount += trainLabel[np.argmin(compareResult)] == testLabel[i]
    accuracy.append(predictRightCount/testSetSize)
plt.figure('Accuracy curve')
plt.plot(accuracy)
plt.xlabel('Projection dimensions')
plt.ylabel('Accuracy')
plt.title('Accuracy curve')
plt.show()
print('Best accuracy: '+str(np.max(accuracy)))  # 98.75% 


# 人脸识别(kNN)
# def kNNClassify(query, features, labels, k):
#     """
#     输入:
#         query--测试样本的特征向量，长度为m
#         features--训练样本的特征向量，shape(n, m)，n为训练样本数
#         labels--对应features顺序的分类标签，长度为n
#         k--kNN的超参数
#     注意:
#         query与features应当是归一化后的特征数据
#     """
#     ## Matrix computing
#     queryExpand = np.tile(query, (features.shape[0], 1))
#     distance = np.sqrt(np.sum((features-queryExpand)**2, axis=1))
#     return np.argmax(np.bincount(np.reshape(labels[np.argsort(distance)[:int(k)]], (-1,))))

# testData = testData - trainDataMean  # 我们手上有的仅仅是训练集上的数据，因此减去训练集的均值向量
# accuracy = []
# for projectDim in range(1, trainSetSize+1):
# # for projectDim in range(1, 39+1):
#     # 训练集中每个人脸都向特征向量进行投影，分别顺序投影至1维和360维特征空间
#     # 得到的trainSetProjections矩阵的行数为投影维度数，列数为训练集的大小
#     trainSetProjections = np.matmul(np.transpose(trainData), eigenVectors[:, :projectDim])
#     testSetProjections = np.matmul(np.transpose(testData), eigenVectors[:, :projectDim])
#     predictRightCount = 0
#     for i in range(testSetSize):
#         testSetProjection = testSetProjections[i, :]
#         compareResult = kNNClassify(testSetProjection, trainSetProjections, np.array(trainLabel), 7)
#         predictRightCount += int(compareResult) == testLabel[i]
#     accuracy.append(predictRightCount/testSetSize)
# plt.figure('Accuracy curve')
# plt.plot(accuracy)
# plt.xlabel('Projection dimensions')
# plt.ylabel('Accuracy')
# plt.title('Accuracy curve')
# plt.show()
# print('Best accuracy: '+str(np.max(accuracy)))  # 97.5%; projectionDims=150






