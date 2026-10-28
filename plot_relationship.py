# https://cloud.tencent.com/developer/article/1752011

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def plot_relationship(conf, fig_save_path):

    # data
    scenes = ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square', 'shopping_mall', 'street_pedestrian', 'street_traffic', 'tram']
    # class_abs = ['indoor', 'outdoor', 'transportation']

    # conf = [[187, 1, 2, 11, 0, 16, 49, 31, 0, 0],
    #         [0, 202, 37, 0, 2, 1, 0, 0, 0, 55],
    #         [3, 8, 215, 8, 2, 1, 3, 0, 3, 54],
    #         [24, 2, 16, 195, 0, 4, 39, 5, 4, 8],
    #         [0, 1, 3, 0, 235, 36, 0, 2, 19, 1],
    #         [4, 0, 0, 5, 29, 215, 3, 10, 31, 0],
    #         [61, 0, 0, 34, 0, 3, 191, 8, 0, 0],
    #         [32, 3, 0, 11, 7, 87, 33, 115, 7, 2],
    #         [0, 0, 0, 3, 8, 34, 0, 2, 250, 0],
    #         [1, 20, 51, 1, 2, 3, 0, 0, 2, 217]]

    # conf = [[ 791, 71, 29],
    #         [91, 1087, 10],
    #         [16, 16, 859]]

    N = len(conf)   # 10

    # --------------------- 1. 创建图 ---------------------
    # G = nx.Graph()          # 无多重边无向图
    G = nx.DiGraph()        # 无多重边有向图
    # G = nx.MultiGraph()     # 有多重边无向图
    # G = nx.MultiDiGraph()   # 有多重边有向图

    # --------------------- 2. 添加节点 ---------------------
    # G.add_node()#添加节点1
    # G.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])    # 添加多个节点
    # G.add_nodes_from(scenes)    # 添加多个节点
    # color=['#7FFF00', '#0000ff', '#9999ff', '#ff00ff', ]
    # RGB r, g, b
    r_ = ['#ff0000', '#ffc125', '#ee7942']
    g_ = ['#00ff00', '#66cd00', '#c0ff3e', '#7cfc00']
    b_ = ['#436eee', '#48d1cc', '#1e90ff']

    if len(conf) == 10:
        color = [r_[0], b_[0], b_[1], r_[1], g_[0], g_[1], r_[2], g_[2], g_[3], b_[2]]
    elif len(conf) == 3:
        color = [r_[0], g_[0], b_[0]]
    else:
        raise ValueError('conf dim error!')

    # node_size
    node_size = []

    # --------------------- 3. 添加边 ---------------------
    # G.add_edge('x', 'y')   # 添加一条边起点为x,终点为y
    # G.add_edge('A', 'B', length = 1) 设置边长度
    # G.add_edges_from([(0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 9)])              # 添加多条边
    # G.add_nodes_from(range(N))
    for i in range(N):  # 添加节点
        G.add_node(i, scene=scenes[i])

    edge_color = []

    for i, data in enumerate(conf):
        node_size.append(data[i]/sum(data[:]) * 1000)       # 设置节点大小
        for j in range(N):
            # if i == j or data[j] == 0:
            # if i == j or data[j] < 5:
            if i == j or data[j] < 10:
            # if i == j or data[j] < 20:
                continue
            G.add_edge(i, j, length=int(data[j]+conf[j][i]))
            edge_color.append(color[i])

    # --------------------- 4. 添节布局属性 ---------------------
    # pos =nx.circular_layout(G)      # 节点在一个圆环上均匀分布
    # pos =nx.random_layout(G)      # 默认，节点随机分布
    # pos =nx.shell_layout(G)       # 节点在同心圆上分布
    pos = nx.spring_layout(G)     # 用Fruchterman-Reingold算法排列节点（样子类似多中心放射状）
    # pos = nx.spectral_layout(G)   # 根据图的拉普拉斯特征向量排列节点

    # --------------------- 5. 画图 ---------------------
    nx.draw(G, with_labels=True, node_color=color, pos=pos, node_size=node_size, edge_color=edge_color, font_size=16)
    # nx.draw(G, with_labels=True, node_color=color, node_size=node_size)

    # --------------------- 显示节点和边的属性 ---------------------
    # https://blog.csdn.net/qq_41854763/article/details/103405760
    # generate node_labels manually
    node_labels = {}
    for node in G.nodes:
        node_labels[node] = G.nodes[node]['scene']  # G.nodes[node] will return all attributes of node
    # generate edge_labels manually
    edge_labels = {}
    for edge in G.edges:
        edge_labels[edge] = G[edge[0]][edge[1]]['length']  # G[edge[0]][edge[1]] will return all attributes of edge

    # nx.draw_networkx_labels(G, pos, labels=node_labels)               # 设置节点标签
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)       # 设置边标签

    # --------------------- 图片保存与展示 ---------------------
    # plt.savefig('relationship_class10_spring.jpg', dpi=600)
    plt.savefig(fig_save_path, dpi=600)
    plt.show()

def main():
    conf = np.loadtxt('two-level_fusion_conf_matrix.txt')

    fig_save_path = 'pictures/relationship_class10_spring.jpg'
    plot_relationship(conf, fig_save_path)
    print(conf)


if __name__ == '__main__':
    main()
