def dfs(G, i, color):
    r = len(G)
    color[i] = -1
    is_DAG = 1
    for j in range(r):
        if G[i][j] != 0:
            # print j
            if color[j] == -1:
                is_DAG = 0
            elif color[j] == 0:
                is_DAG = dfs(G, j, color)
    color[i] = 1
    return is_DAG


def findcircle(G):
    r = len(G)
    color = [0] * r
    for i in range(r):
        # print i
        if color[i] == 0:
            is_DAG = dfs(G, i, color)
            if is_DAG == 0:
                break
    return is_DAG


G = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
is_DAG = findcircle(G)
print(is_DAG)
