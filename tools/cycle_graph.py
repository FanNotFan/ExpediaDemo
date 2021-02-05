import pandas as pd

class DFSFindCircle(object):
    trace = []
    visited = []
    has_circle = False

    def __init__(self, inputDataFrame=None, name='.'):
        self.name = name
        self.inputDataFrame = inputDataFrame
        self.index_list = inputDataFrame.index.tolist()
        self.index_length = len(inputDataFrame.index.tolist())

    def dfs(self, i, color):
        color[i] = -1
        has_circle = False
        for j, val in enumerate(self.index_list):# 遍历当前节点i的所有邻居节点
            if self.inputDataFrame[self.index_list[i]][self.index_list[j]]:
                self.visited.append((self.index_list[j],self.index_list[i]))
                if color[j] == -1:
                    has_circle = True
                    if len(self.trace) < len(self.visited):
                        self.trace = self.visited.copy()
                    self.visited = []
                elif color[j] == 0:
                    has_circle = self.dfs(j,color)
        color[i] = 1
        return has_circle

    def findcircle(self):
        # color = 0 该节点暂未访问
        # color = -1 该节点访问了一次
        # color = 1 该节点的所有孩子节点都已访问,就不会再对它做DFS了
        color = [0] * self.index_length
        has_circle = True
        for i, val in enumerate(self.index_list):
            if color[i] == 0:
                has_circle = self.dfs(i,color)
                if has_circle == False:
                    break
        return has_circle

if __name__ == '__main__':
    a = [[False,True,False],[False,False,True],[True,False,False]]  #这里的1说明行index对应的节点指向列index对应的节点，对角线处为0
    # G = [[False,False,False,True,False],[True,False,False,False,False],[False,False,False,True,True],[False,False,False,False,False],[False,True,False,False,False]]
    df = pd.DataFrame(a)
    dfs = DFSFindCircle(inputDataFrame=df)
    has_circle = dfs.findcircle()
    print(has_circle)
    if has_circle:
        print(dfs.trace)
