import pandas as pd
import numpy as np
import xlwt


sheet_list = ['00_haisang', '01_qiuqie', '02_tonghua', '03_mulan', '04_lujue', '05_laoshule', '06_haiqi', '07_yinyeshu',
              '08_huangjin', '09_luwei', '10_water', '11_other']
all_train = []
all_test = []

for sheetnames in sheet_list:
    original_data = pd.read_excel(r".xlsx",
                                  sheet_name=sheetnames)

    x = original_data.iloc[0:, 7:456].values  # 特征
    y = original_data.iloc[0:, 0].values  # 标签
    test_size = 0.4


    # :param x: shape (n_samples, n_features)
    # :param y: shape (n_sample, )
    # :param test_size: the ratio of test_size (float)
    # :return: spec_train: (n_samples, n_features)
    #          spec_test: (n_samples, n_features)
    #          target_train: (n_sample, )
    #          target_test: (n_sample, )

    M = x.shape[0]
    N = round((1 - test_size) * M)
    samples = np.arange(M)

    D = np.zeros((M, M))

    for i in range((M - 1)):
        xa = x[i, :]
        for j in range((i + 1), M):
            xb = x[j, :]
            D[i, j] = np.linalg.norm(xa - xb)

    maxD = np.max(D, axis=0)
    index_row = np.argmax(D, axis=0)
    index_column = np.argmax(maxD)

    m = np.zeros(N)
    m[0] = np.array(index_row[index_column])
    m[1] = np.array(index_column)
    m = m.astype(int)
    dminmax = np.zeros(N)
    dminmax[1] = D[m[0], m[1]]

    for i in range(2, N):
        pool = np.delete(samples, m[:i])
        dmin = np.zeros((M - i))
        for j in range((M - i)):
            indexa = pool[j]
            d = np.zeros(i)
            for k in range(i):
                indexb = m[k]
                if indexa < indexb:
                    d[k] = D[indexa, indexb]
                else:
                    d[k] = D[indexb, indexa]
            dmin[j] = np.min(d)
        dminmax[i] = np.max(dmin)
        index = np.argmax(dmin)
        m[i] = pool[index]

    m_complement = np.delete(np.arange(x.shape[0]), m)
    print('训练数据：', m)
    print('测试数据：', m_complement)
    train_data = original_data.iloc[m, :]
    test_data = original_data.iloc[m_complement, :]
    print(train_data)
    print(test_data)
    all_train.append(train_data)
    all_test.append(test_data)

with pd.ExcelWriter(r"train_data.xlsx") as trainwriter:
    for i, j in enumerate(sheet_list):
        all_train[i].to_excel(trainwriter, sheet_name=j)

    trainwriter.save()
    trainwriter.close()


with pd.ExcelWriter(r"test_data.xlsx") as testwriter:
    for i, j in enumerate(sheet_list):
        all_test[i].to_excel(testwriter, sheet_name=j)

    testwriter.save()
    testwriter.close()



print('样本划分完成')


