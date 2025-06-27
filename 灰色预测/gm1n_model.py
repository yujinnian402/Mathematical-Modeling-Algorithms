import numpy as np

def GM1N(x0, y0):
    x0, y0 = np.array(x0), np.array(y0)
    n, m = y0.shape # 获取y0数组的形状,n表示y0的行数(样本数量等),m表示列数(变量维度等)
    x1 = x0.cumsum()
    z1 = 0.5 * (x1[1:] + x1[:-1])
    Y = x0[1:] # 因变量序列Y
    B = np.hstack([-z1.reshape(-1, 1), y0[1:], np.ones((n-1, 1))]) # z1.reshape(-1,1)将z1转换为列向量(-1表示自动计算行数,1表示列数为1)
       # y0[1:]:截取y0从第2个样本开始的部分,作为多变量输入
    params = np.linalg.lstsq(B, Y, rcond=None)[0] # 求解模型参数params
    a, b_list, c = params[0], params[1:-1], params[-1] # 从params中解析出a(发展系数),b_list(对应多变量的驱动系数,长度由y0列数决定),c(常数项)
    def predict(k, y_ex): # k:预测步数    y_ex:外部输入的多变量预测辅助值
        return (x0[0] - (np.dot(b_list, y_ex)+c)/a) * np.exp(-a*k) + (np.dot(b_list, y_ex)+c)/a
    return predict

if __name__ == "__main__":
    x0 = [100, 105, 110, 115, 122, 130] # 主序列数据
    y0 = [[30, 50], [33, 54], [35, 58], [39, 63], [43, 68], [46, 72]] # 多变量辅助序列数据
    predict_func = GM1N(x0, np.array(y0))
    print("预测下一期主变量：", predict_func(6, [48, 75]))
