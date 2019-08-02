import numpy  as np

def Solve(maps, M):
    res = np.zeros_like (maps)
    if M == 1:
        return maps
    res_last = Solve(maps, M-1)
    N = maps.shape[0]
    for i in range (N):
        for j in range (i, N):
            res[i,j] = MinPath (res_last, maps, i, j, M)
            res[j,i] = res[i,j]
    return res
def MinPath(res_last, maps, i, j, M):
    temp_line = res_last[i,:] + maps[:,j]
    if M == 2:
        temp_line[i] = 1e8
        temp_line[j] = 1e8
    else:
        temp_line[j] = 1e8
    return np.min(temp_line)
if __name__ == '__main__':
    # maps = np.array([[0,2,3],[2,0,1],[3,1,0]])
    maps = np.array ([[0, 2, 3],
                      [2, 0, 1],
                      [3, 1, 0],
                      ])
    N = int(raw_input ("N="))
    M = int(raw_input ("M="))
    maps = list(raw_input("maps="))
    result = Solve(maps, M)
    print(result)