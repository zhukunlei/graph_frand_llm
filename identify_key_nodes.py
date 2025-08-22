import pandas as pd
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
import json
from prompt_create import build_graph, identify_key_nodes

if __name__ == "__main__":
    data_path = "graph_synthetic_data.txt"
    G, data = build_graph(data_path)
    df_metrics = identify_key_nodes(G)
    df_metrics.to_csv("key_nodes.csv", encoding="utf-8")
    nx.write_gpickle(G, "graph.pkl")  # 保存图给下一步用
    print("重要节点识别完成，结果已保存到 key_nodes.csv")