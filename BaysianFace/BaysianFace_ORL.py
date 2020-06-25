import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 由于Baysian Face Recognition涉及随机选取‘intrapersonal对’和‘extrapersonnel对’
# 因此需要设定随机数种子，这里使用time返回的数字作为种子
import time
np.random.seed(int(time.time()))

# ORL数据集一共有400张图片，在这里可以分配训练集和测试集的大小
# trainImgsPerPerson用以表示数据集中每个人的10张照片中有多少张作为训练集
trainSetSize = 320
testSetSize = 80
trainImgsPerPerson = trainSetSize//40
assert trainSetSize + testSetSize == 400


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
    输出训练集和测试集的图像路径list，以及对应的label，顺序是一一对应的
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

# 经过上述处理，终于拿到原始的图像数据了
# 但是Baysian Face Recognition训练时需要的数据还需进一步获取，也就是intrapersonal和extrapersonnal的delta数据
# intraDifferPerPerson是指对于每个subject，由其对应的训练数据中，获取几个intrapersonnel的delta
# 这样做的原因是为了不对某个subject偏心，每个subject都提供intraDifferPerPerson个delta_intra数据
# 这样一来，intrapersonnalTrainingSamplesNum就是所有的delta_intra的个数
# extrapersonnalTrainingSamplesNum即delta_extra的个数
intraDifferPerPerson = 6
intrapersonnalTrainingSamplesNum = int(40*intraDifferPerPerson)  # each person 2
extrapersonnalTrainingSamplesNum = 240


def getTrainingDeltaIntra(trainData, trainLabel):
    '''
    输入：训练数据和对应的label
    输出：intrapersonnel的delta，依照训练数据的格式，每一列为一个delta
    '''
    deltaIntra = np.zeros((112*92, intrapersonnalTrainingSamplesNum))
    for i in range(40):
        # 这里对每个subject进行循环
        # 获取delta_intra的策略是：在每个subject对应的训练数据列中随机选取两个不同列做差得到delta
        # 随机选取的两列的下标为firstImageIndex和secondImageIndex，其范围为[0, trainImgsPerPerson]
        # 这里我做了两个处理来保证策略能够成功运行：
        # 一是在得到firstImageIndex后，选取secondImageIndex时，要确保secondImageIndex与firstImageIndex不同
        # 二是对于每个已经用过的firstImageIndex都进行记录，确保其不会再出现，这样设置主要用于每个subject的训练数据较少的情况，可以根据情况不使用该处理方法
        usedFirstImageIndex = []
        for j in range(intraDifferPerPerson):
            firstImageIndex = np.random.randint(0, trainImgsPerPerson)
            while firstImageIndex in usedFirstImageIndex:
                firstImageIndex = np.random.randint(0, trainImgsPerPerson)
            usedFirstImageIndex.append(firstImageIndex)
            secondImageIndex = np.random.randint(0, trainImgsPerPerson)
            while secondImageIndex - firstImageIndex == 0:
                secondImageIndex = np.random.randint(0, trainImgsPerPerson)
            # 为了保险，这里确保选取的两列训练数据对应的label是一致的
            assert trainLabel[i*trainImgsPerPerson+firstImageIndex] == trainLabel[i*trainImgsPerPerson+secondImageIndex]
            deltaIntra[:, i*intraDifferPerPerson+j] = trainData[:, i*trainImgsPerPerson+firstImageIndex]-trainData[:, i*trainImgsPerPerson+secondImageIndex]
    return deltaIntra


def getTrainingDeltaExtra(trainData, trainLabel):
    '''
    输入：训练数据和对应的label
    输出：extrapersonnel的delta，依照训练数据的格式，每一列为一个delta
    '''
    deltaExtra = np.zeros((112*92, extrapersonnalTrainingSamplesNum))
    for i in range(40):
        # 同样地，为了不偏心，对每个subject进行循环，也就是说每个subject的图像都会和其他subject做差，且概率相同
        # 这里我获取delta_extra使用的策略是：
        # 首先在每个subject自身对应的训练数据中随机挑选一张，再在非自身的subject里任选一个subject，再从该subject的训练数据中任选一个，做差得到delta
        # firstImageIndex即自身对应的训练数据中随机挑选的那一张的列数（每一列为一条样本数据）
        # secondPeopleIndex为非自身的subject里任选的那一个subject，一共40个人，在[0,39]中任选一个整数，且需要保证不与当前subject相同
        # secondImageIndex即该非自身subject的训练数据中任选的一条训练数据的下标
        for j in range(extrapersonnalTrainingSamplesNum//40):
            firstImageIndex = np.random.randint(0, trainImgsPerPerson)
            secondPeopleIndex = np.random.randint(0, 40)
            while secondPeopleIndex - i == 0:
                secondPeopleIndex = np.random.randint(0, 40)
            secondImageIndex = np.random.randint(0, trainImgsPerPerson)
            # 使用断言，保证拿到的两列数据属于两个不同的subject
            assert trainLabel[i*trainImgsPerPerson+firstImageIndex] != trainLabel[secondPeopleIndex*trainImgsPerPerson+secondImageIndex]
            deltaExtra[:, i*(extrapersonnalTrainingSamplesNum//40)+j] = trainData[:, i*trainImgsPerPerson+firstImageIndex]-trainData[:, secondPeopleIndex*trainImgsPerPerson+secondImageIndex]
    return deltaExtra

# 使用上面的两个函数得到deltaIntra和deltaExtra
deltaIntra = getTrainingDeltaIntra(trainData, trainLabel)
deltaExtra = getTrainingDeltaExtra(trainData, trainLabel)


def calculateIntraSpaceProbability(deltaProjectOnIntra, eigenValuesIntra):
    '''
    输入：测试时计算得到的delta_intra在deltaIntra主空间各个特征向量上的投影值；deltaIntra主空间各个特征向量对应的特征值
    输出：根据论文中的公式计算得到delta_intra落在deltaIntra主空间上的似然概率
    '''
    eigenValuesIntra = eigenValuesIntra[:MIntra]
    return np.exp(-0.5*np.sum(deltaProjectOnIntra**2/eigenValuesIntra))/np.prod(np.power(eigenValuesIntra, 1/2))/np.power(2*np.pi, MIntra/2)

def calculateExtraSpaceProbability(deltaProjectOnExtra, eigenValuesExtra):
    '''
    输入：测试时计算得到的delta_extra在deltaExtra主空间各个特征向量上的投影值；deltaExtra主空间各个特征向量对应的特征值
    输出：根据论文中的公式计算得到delta_extra落在deltaExtra主空间上的似然概率
    '''
    eigenValuesExtra = eigenValuesExtra[:MExtra]
    return np.exp(-0.5*np.sum(deltaProjectOnExtra**2/eigenValuesExtra))/np.prod(np.power(eigenValuesExtra, 1/2))/np.power(2*np.pi, MExtra/2)

# 设置往deltaIntra主空间投影的维数为MIntra，往deltaExtra主空间投影的维数为MExtra
MIntra = 40
MExtra = 40
assert MIntra < intrapersonnalTrainingSamplesNum
assert MExtra < extrapersonnalTrainingSamplesNum
# 使用数学方法简化计算协方差矩阵的特征值和特征向量的运算复杂度，计算得到deltaIntra和deltaExtra协方差矩阵的特征值和特征向量
intraDataCovarianceMatrix = np.matmul(np.transpose(deltaIntra), deltaIntra)/intrapersonnalTrainingSamplesNum
eigenValuesIntra, eigenVectorsIntra = np.linalg.eig(intraDataCovarianceMatrix)
eigenVectorsIntra = np.matmul(deltaIntra, eigenVectorsIntra)
eigenVectorsIntra = np.real(eigenVectorsIntra[:, :MIntra])
eigenVectorsIntra = eigenVectorsIntra/np.linalg.norm(eigenVectorsIntra, axis=0, keepdims=True)
extraDataCovarianceMatrix = np.matmul(np.transpose(deltaExtra), deltaExtra)/extrapersonnalTrainingSamplesNum
eigenValuesExtra, eigenVectorsExtra = np.linalg.eig(extraDataCovarianceMatrix)
eigenVectorsExtra = np.matmul(deltaExtra, eigenVectorsExtra)
eigenVectorsExtra = np.real(eigenVectorsExtra[:, :MExtra])
eigenVectorsExtra = eigenVectorsExtra/np.linalg.norm(eigenVectorsExtra, axis=0, keepdims=True)

# 人脸识别
# predictRightCount用以记录预测准确的次数
predictRightCount = 0
for i in range(testSetSize):
    # 取出一条测试数据testFace
    # similarProbilities用以记录intrapersonnel的似然概率
    # 具体来说，对于每一条测试数据，它需要和训练集中的每一条数据做差，并往deltaIntra主空间上投影，根据投影值计算似然概率
    # 最终选取概率值最大的那一条对应的label作为结果，判断与测试数据对应的label是否相同
    testFace = testData[:, i]
    similarProbilities = []
    for j in range(trainSetSize):
        delta = testFace - trainData[:, j]
        deltaProjectOnIntra = np.matmul(np.transpose(delta), eigenVectorsIntra)
        deltaProjectOnExtra = np.matmul(np.transpose(delta), eigenVectorsExtra)
        similarProbilities.append(calculateIntraSpaceProbability(deltaProjectOnIntra, eigenValuesIntra))
    predictRightCount += trainLabel[np.argmax(similarProbilities)] == testLabel[i]
accuracy= predictRightCount/testSetSize
print('Accuracy: '+str(accuracy))

# 0.975 实验证明，MIntra的选择比MExtra的选择重要多了，MIntra几乎决定了最终性能