# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 08:54:35 2018

@author: zy
"""

import numpy  as np


def Solve(maps, M):
    '''
    由N个节点两两连接组成路径，选取从节点i->节点j之间的最短M条路径

    param:
        maps:二维数组，由两两节点之间的路径长度组成
        M:表示所经过的路径个数
    '''
    '''
    输入校验
    '''
    if not isinstance (maps, (list, np.ndarray)):
        raise ValueError ('输入参数maps数据类型必须是list或者numpy.array')

    if len (maps.shape) != 2:
        raise ValueError ('输入参数maps为二维数组')

    if maps.shape[0] != maps.shape[1]:
        raise ValueError ('输入二维数组maps行数和列数要求一致')

    # 计算节点的个数
    N = maps.shape[0]

    if N < 2 or N > 100:
        raise ValueError ('输入二维数组maps行数必须在2~100之间')

    if M < 2 or M > 1E6:
        raise ValueError ('输入参数N的值必须在2~1e6之间')

    # 输入二维数组数值校验
    for i in range (N):
        for j in range (i, N):
            if maps[i][j] != maps[j][i]:
                raise ValueError ('输入二维数组maps必须是对称的')
            if maps[i][j] < -1e8 or maps[i][j] > 1e8:
                raise ValueError ('二维数组maps的元素值必须在-1e8~1e8之间')
            if i == j:
                if maps[i][j] != 0:
                    raise ValueError ('二维数组maps的对角元素值必须是0')

    # 用于保存i->j的路径值
    res = np.zeros_like (maps)

    # 计算节点i->j的最短路径
    for i in range (N):
        for j in range (i, N):
            res[i][j] = MinPath (maps, M, i, j)
            res[j][i] = res[i][j]
    return res


def MinPath(maps, M, i, j):
    '''
    计算i->j的最短路径
    '''
    # 递归终止条件
    if M == 1:
        return maps[i][j]
    '''计算i->j的最短路径'''
    N = maps.shape[0]
    # 用于保存i->j的可能路径长度
    length = np.zeros (N)
    # 遍历从k->j的最短路径
    for k in range (N):
        if k != i and k != j:
            # k->j的M-1条最短路径 + i->k的一条路径
            length[k] = MinPath (maps, M - 1, k, j) + maps[i][k]
            # length[k] = MinPath (maps, M - 1, i, k) + maps[k][j]

    # 进行排序，过滤掉为0的值
    length = np.sort (length)
    for i in length:
        if i != 0:
            return i


if __name__ == '__main__':
    # maps = np.array([[0,2,3],[2,0,1],[3,1,0]])
    maps = np.array ([[0, 2, 3],
                      [2, 0, 1],
                      [3, 1, 0],
                      ])
    M = 2
    result = Solve (maps, M)
    print(result)