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
            if j < 10 - trainImgsPerPerson:
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

# 使用PCA获取原始训练数据的主特征空间对应的特征向量，使用数学技巧进行转化以降低计算复杂度
trainDataMean = np.mean(trainData, axis=1, keepdims=True)
trainData = trainData - trainDataMean
trainDataCovarianceMatrix = np.matmul(np.transpose(trainData), trainData)/trainSetSize
eigenValuesPCA, eigenVectorsPCA = np.linalg.eig(trainDataCovarianceMatrix)
eigenVectorsPCA = np.matmul(trainData, eigenVectorsPCA)
eigenVectorsPCA = eigenVectorsPCA/np.linalg.norm(eigenVectorsPCA, axis=0, keepdims=True)  # shape: (feature number, main feature number)

# 将训练数据降维至设定维数的主特征空间中，设定的维数为mainFeatureNum
mainFeatureNum = 45
trainDataLDA = np.transpose(np.matmul(np.transpose(trainData), eigenVectorsPCA[:, :mainFeatureNum]))  # shape: (main feature num, trainFace num)

# 根据公式计算类间和类内散列矩阵
Sw = np.zeros((mainFeatureNum,mainFeatureNum))
miu = np.mean(trainDataLDA, axis=1, keepdims=True)
Sb = np.zeros((mainFeatureNum,mainFeatureNum))
for i in range(40):
    miu_i = np.mean(trainDataLDA[:, i*trainImgsPerPerson:(i+1)*trainImgsPerPerson], axis=1, keepdims=True)
    Sw += np.matmul(trainDataLDA[:, i*trainImgsPerPerson:(i+1)*trainImgsPerPerson] - miu_i, np.transpose(trainDataLDA[:, i*trainImgsPerPerson:(i+1)*trainImgsPerPerson] - miu_i))
    Sb += trainImgsPerPerson*np.matmul(miu_i-miu, (miu_i-miu).T)

# 根据公式计算FisherFace的投影向量
eigenValuesLDA, eigenVectorsLDA = np.linalg.eig(np.matmul(np.linalg.inv(Sw), Sb))
# 由矩阵的秩的性质可知，计算得到的eigenVectorsLDA中只有前C-1个是有效的，C是类别数，因此仅取前39个。另外，由于计算出来的特征向量含有虚数（这是由于只有前C-1个特征向量有效的而导致的），这就使得即便是纯实数，也带有一个0j的虚数部，所以使用np.real将其转化为实数
eigenValuesLDA, eigenVectorsLDA = eigenValuesLDA[:39], eigenVectorsLDA[:, :39]
eigenVectorsLDA = np.real(eigenVectorsLDA)
# 将先前的PCA降维投影矩阵和现在的FisherFace投影矩阵相乘
eigenVectors = np.matmul(eigenVectorsPCA[:, :mainFeatureNum], eigenVectorsLDA)
# 归一化投影向量，确保可度量性
eigenVectors = eigenVectors/np.linalg.norm(eigenVectors, axis=0, keepdims=True)

# 可视化Fisher脸
for i in range(39):
    fisherFace = eigenVectors[:, i]
    # 可视化之前需要对数据进行归一化处理，方可正常显示
    fisherFace = (fisherFace - np.min(fisherFace))/(np.max(fisherFace)-np.min(fisherFace))
    fisherFace = np.resize(fisherFace, (112, 92))

    # 使用pyplot能够得到热图化输出，更加好看
    plt.figure('fisherFace')
    plt.imshow(fisherFace)
    plt.axis('off')
    plt.show()
    plt.imsave('./fisherFace'+str(i).zfill(2)+'.png', fisherFace)

# 人脸重建
# 随机选取一张人脸图像，读取并灰度化，保存该原始图像
faceChooseToReconstruct = facesImgPath[np.random.randint(0, 400)]
faceChooseToReconstruct = cv2.imread(faceChooseToReconstruct)
faceChooseToReconstruct = cv2.cvtColor(faceChooseToReconstruct, cv2.COLOR_BGR2GRAY)
cv2.imwrite('./originFace.png', faceChooseToReconstruct)
# 将原始图像拉成一个长向量并减去均值
faceChooseToReconstruct = np.array(faceChooseToReconstruct.flatten(), dtype=np.float)-trainDataMean.flatten()
# 将原始图像分别投影至projectDims维度数的特征子空间中
projectDims = 39
for dim in range(projectDims):
    projectValues = np.matmul(np.transpose(faceChooseToReconstruct), eigenVectors[:,:dim+1])
    reconstructFace = np.zeros(112*92)
    # 使用投影值，以及每个投影值对应的特征向量进行人脸重建
    for i in range(dim+1):
        reconstructFace += projectValues[i]*eigenVectors[:, i]
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
testData = testData - trainDataMean  # 我们手上有的仅仅是训练集上的数据，因此减去训练集的均值向量
accuracy = []
for projectDim in range(1, 40):
    # 训练集中每个人脸都向特征向量进行投影，分别顺序投影至1维和39维特征空间
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
print('Best accuracy: '+str(np.max(accuracy)))  # 100% 这里面主要需要调整的是刚开始PCA的投影维度，该投影维度极大地影响着性能



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
# for projectDim in range(1, 40):
#     # 训练集中每个人脸都向特征向量进行投影，分别顺序投影至1维和39维特征空间
#     # 得到的trainSetProjections矩阵的行数为投影维度数，列数为训练集的大小
#     trainSetProjections = np.matmul(np.transpose(trainData), eigenVectors[:, :projectDim])
#     testSetProjections = np.matmul(np.transpose(testData), eigenVectors[:, :projectDim])
#     predictRightCount = 0
#     for i in range(testSetSize):
#         testSetProjection = testSetProjections[i, :]
#         compareResult = kNNClassify(testSetProjection, trainSetProjections, np.array(trainLabel), 1)
#         predictRightCount += int(compareResult) == testLabel[i]
#     accuracy.append(predictRightCount/testSetSize)
# plt.figure('Accuracy curve')
# plt.plot(accuracy)
# plt.xlabel('Projection dimensions')
# plt.ylabel('Accuracy')
# plt.title('Accuracy curve')
# plt.show()
# print('Best accuracy: '+str(np.max(accuracy)))  # 97.5%; projectionDims=21