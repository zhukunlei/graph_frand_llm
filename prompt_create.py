import networkx as nx
import pandas as pd
import numpy as np
import json
from community import community_louvain
from sklearn.preprocessing import MinMaxScaler
import warnings
from collections import ChainMap
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import  torch
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch.nn import functional as F





warnings.filterwarnings('ignore')

cols_dict = {'user_id': '用户标识符', 'date': '日期', 'year': '年份', 'label': '是否是欺诈用户',
             'opp_user_id': '通话对端用户', 'CALL_CNT': '通话次数', 'CALLED_CNT': '被叫通话次数',
             'TOTAL_CNT': '总通话次数', 'CALL_TIME_LENGTH': '主叫通话时长', 'CALLED_TIME_LENGTH': '被叫通话时长',
             'TOTAL_TIME_LENGTH': '总通话时长 ', 'ONLINE_DUR': 'app使用时长', 'QUERY_CNT': '使用次数',
             'QUERY_DAYS': '使用天数', 'APP_CNT': '访问app次数', 'GENDER': '性别', 'AGE': '年龄',
             'CUST_STAR_LVL': '星级', 'IS_ENT': '是否与集团相关', 'INNET_MONTHS': '在网月份',
             'PRODUCT_TYPE': '套餐类型', 'TERM_TYPE': '终端类型', 'TERM_BRAND': '终端品牌',
             'THIS_ACCT_FEE_TAX': '当月消费总金额 ', 'TOTAL_CALL_CNT_1M': '近1月总通话次数 ',
             'TOTAL_CALL_CNT_3M_SUM': '近3月总通话次数 ', 'TOTAL_CALL_CNT_6M_SUM': '近6月总通话次数 ',
             'TOTAL_CALL_CNT_3M_MEAN': '近3月平均通话次数', 'TOTAL_CALL_CNT_6M_MEAN': '近6月平均通话次数',
             'CALLED_CNT_1M': '近1月呼入通话次数 ', 'CALLED_CNT_3M_SUM': '近3月呼入通话次数 ',
             'CALLED_CNT_6M_SUM': '近6月呼入通话次数 ', 'CALL_CNT_3M_MEAN': '近三个月的平均通话次数',
             'CALL_CNT_6M_MEAN': '近六个月的平均通话次数', 'CALL_OPS_NBR_CNT_1M': '近1月不重复通话对端数的加和 ',
             'CALL_OPS_NBR_CNT_3M_SUM': '近3月不重复通话对端数的加和',
             'CALL_OPS_NBR_CNT_6M_SUM': '近6月不重复通话对端数的加和 ', 'IMSG_MT_SMS_QTY_1M': '近1月行业短信接收量 ',
             'IMSG_MT_SMS_QTY_3M_SUM': '近3月行业短信接收量 ', 'IMSG_MT_SMS_QTY_6M_SUM': '近6月行业短信接收量 ',
             'IN_BANK_CNT': '当月接听国有大行电话的次数 ', 'IN_STOCK_BANK_CNT': '当月接听股份制银行电话的次数 ',
             'IN_PL_BANK_CNT': '当月接听民营银行电话的次数 ', 'IN_CONS_FIN_CNT': '当月接听重点消费金融公司电话的次数 ',
             'BANK_SMS_ACCT_CNT': '当月接收国有大行动账短信条数 ',
             'STOCK_BANK_SMS_SCCT_CNT': '当月接收股份制银行动账短信条数 ',
             'PL_BANK_SMS_ACCT_CNT': '当月接收民营银行动账短信条数 ', 'BANK_SMS_CNT': '当月接收国有大行短信条数 ',
             'STOCK_BANK_SMS_CNT': '当月接收股份制银行短信条数 ', 'PL_BANK_SMS_CNT': '当月接收民营银行短信条数 ',
             'CONS_FIN_SMS_CNT': '当月接收重点消费金融公司短信条数 ', 'PAY_RT_1M': '近1月缴费金额的加和(元) ',
             'PAY_RT_3M_SUM': '近3月缴费金额的加和(元) ', 'PAY_RT_6M_SUM': '近6月缴费金额的加和(元) ',
             'TOT_TAOCAN_FEE_1M': '近1月套餐消费总金额(元) ', 'TOT_TAOCAN_FEE_3M_SUM': '近3月套餐消费总金额(元)',
             'TOT_TAOCAN_FEE_6M_SUM': '近6月套餐消费总金额(元)', 'OWE_FEE_1M': '近1月欠费总金额(元)',
             'OWE_FEE_3M_SUM': '近3月欠费总金额(元) ', 'OWE_FEE_6M_SUM': '近6月欠费总金额(元)',
             'TOT_FEE_1M': '近1月消费金额(元) ', 'TOT_FEE_3M_SUM': '近3月消费金额(元) ',
             'TOT_FEE_6M_SUM': '近6月消费金额(元) ', 'OWE_CNT_1M': '近1月余额为负次数 ',
             'OWE_CNT_3M': '近3月余额为负次数', 'OWE_CNT_6M': '近6月余额为负次数 '}
feature_desc_dict = {'ONLINE_DUR': 'app使用时长', 'QUERY_CNT': '使用次数', 'QUERY_DAYS': '使用天数',
                     'APP_CNT': '访问app次数', 'GENDER': '性别', 'AGE': '年龄', 'CUST_STAR_LVL': '星级',
                     'IS_ENT': '是否与集团相关', 'INNET_MONTHS': '在网月份', 'PRODUCT_TYPE': '套餐类型',
                     'TERM_TYPE': '终端类型', 'TERM_BRAND': '终端品牌', 'THIS_ACCT_FEE_TAX': '当月消费总金额 ',
                     'TOTAL_CALL_CNT_1M': '近1月总通话次数 ', 'TOTAL_CALL_CNT_3M_SUM': '近3月总通话次数 ',
                     'TOTAL_CALL_CNT_6M_SUM': '近6月总通话次数 ', 'TOTAL_CALL_CNT_3M_MEAN': '近3月平均通话次数',
                     'TOTAL_CALL_CNT_6M_MEAN': '近6月平均通话次数', 'CALLED_CNT_1M': '近1月呼入通话次数 ',
                     'CALLED_CNT_3M_SUM': '近3月呼入通话次数 ', 'CALLED_CNT_6M_SUM': '近6月呼入通话次数 ',
                     'CALL_CNT_3M_MEAN': '近三个月的平均通话次数', 'CALL_CNT_6M_MEAN': '近六个月的平均通话次数',
                     'CALL_OPS_NBR_CNT_1M': '近1月不重复通话对端数的加和 ',
                     'CALL_OPS_NBR_CNT_3M_SUM': '近3月不重复通话对端数的加和',
                     'CALL_OPS_NBR_CNT_6M_SUM': '近6月不重复通话对端数的加和 ',
                     'IMSG_MT_SMS_QTY_1M': '近1月行业短信接收量 ', 'IMSG_MT_SMS_QTY_3M_SUM': '近3月行业短信接收量 ',
                     'IMSG_MT_SMS_QTY_6M_SUM': '近6月行业短信接收量 ', 'IN_BANK_CNT': '当月接听国有大行电话的次数 ',
                     'IN_STOCK_BANK_CNT': '当月接听股份制银行电话的次数 ',
                     'IN_PL_BANK_CNT': '当月接听民营银行电话的次数 ',
                     'IN_CONS_FIN_CNT': '当月接听重点消费金融公司电话的次数 ',
                     'BANK_SMS_ACCT_CNT': '当月接收国有大行动账短信条数 ',
                     'STOCK_BANK_SMS_SCCT_CNT': '当月接收股份制银行动账短信条数 ',
                     'PL_BANK_SMS_ACCT_CNT': '当月接收民营银行动账短信条数 ',
                     'BANK_SMS_CNT': '当月接收国有大行短信条数 ', 'STOCK_BANK_SMS_CNT': '当月接收股份制银行短信条数 ',
                     'PL_BANK_SMS_CNT': '当月接收民营银行短信条数 ',
                     'CONS_FIN_SMS_CNT': '当月接收重点消费金融公司短信条数 ', 'PAY_RT_1M': '近1月缴费金额的加和(元) ',
                     'PAY_RT_3M_SUM': '近3月缴费金额的加和(元) ', 'PAY_RT_6M_SUM': '近6月缴费金额的加和(元) ',
                     'TOT_TAOCAN_FEE_1M': '近1月套餐消费总金额(元) ',
                     'TOT_TAOCAN_FEE_3M_SUM': '近3月套餐消费总金额(元)',
                     'TOT_TAOCAN_FEE_6M_SUM': '近6月套餐消费总金额(元)', 'OWE_FEE_1M': '近1月欠费总金额(元)',
                     'OWE_FEE_3M_SUM': '近3月欠费总金额(元) ', 'OWE_FEE_6M_SUM': '近6月欠费总金额(元)',
                     'TOT_FEE_1M': '近1月消费金额(元) ', 'TOT_FEE_3M_SUM': '近3月消费金额(元) ',
                     'TOT_FEE_6M_SUM': '近6月消费金额(元) ', 'OWE_CNT_1M': '近1月余额为负次数 ',
                     'OWE_CNT_3M': '近3月余额为负次数', 'OWE_CNT_6M': '近6月余额为负次数'}


# 1. 数据准备与图谱构建
def build_graph(data_path):
    """构建通话社交网络图"""
    cols = list(cols_dict.keys())
    data = pd.read_table(data_path, sep='|', header=None, names=cols, encoding='utf8')

    # 创建有向图
    G = nx.DiGraph()

    # 添加边和属性
    for _, row in data.iterrows():
        G.add_edge(row['user_id'], row['opp_user_id'],
                   calls=row['TOTAL_CNT'],
                   duration=row['TOTAL_TIME_LENGTH'],
                   weight=row['TOTAL_CNT'] * 0.5 + row['TOTAL_TIME_LENGTH'] * 0.01)

    # 为节点添加 label 属性
    label_dict = data.set_index('user_id')['label'].to_dict()

    # 为节点添加联系对象信息
    contact_dict = data.groupby('user_id')['opp_user_id'].apply(list).to_dict()

    for node in G.nodes():
        if node in label_dict:
            G.nodes[node]['label'] = label_dict[node]
        else:
            G.nodes[node]['label'] = 0  # 默认值

        # 添加联系人信息
        G.nodes[node]['contacts'] = contact_dict.get(node, [])

        # 添加原始特征
        if node in data['user_id'].values:
            user_data = data[data['user_id'] == node].iloc[0]
            for key in feature_desc_dict.keys():
                G.nodes[node][key] = user_data[key]

    return G, data


# 2. 重要节点识别
def identify_key_nodes(G):
    """识别网络中的重要节点"""
    metrics = {}
    metrics['degree'] = nx.degree_centrality(G)
    metrics['betweenness'] = nx.betweenness_centrality(G)
    metrics['pagerank'] = nx.pagerank(G)

    # 合并所有指标
    df_metrics = pd.DataFrame(metrics)

    # 归一化
    scaler = MinMaxScaler()
    df_metrics_scaled = pd.DataFrame(scaler.fit_transform(df_metrics),
                                     columns=df_metrics.columns,
                                     index=df_metrics.index)

    # 计算综合重要性评分
    df_metrics_scaled['importance'] = (df_metrics_scaled['degree'] * 0.2 +
                                       df_metrics_scaled['betweenness'] * 0.3 +
                                       df_metrics_scaled['pagerank'] * 0.5)

    return df_metrics_scaled.sort_values('importance', ascending=False)


# 3. 社区发现与分析
def community_detection(G):
    """社区发现与社区分析"""
    # 转换为无向图
    G_undirected = G.to_undirected()
    partition = community_louvain.best_partition(G_undirected)

    # 添加社区属性到节点
    for node in G.nodes():
        G.nodes[node]['community'] = partition.get(node, -1)

    # 计算社区欺诈比例
    community_fraud_ratio = {}
    communities = {}
    for node, comm in partition.items():
        if comm not in communities:
            communities[comm] = []
        communities[comm].append(node)

    # 计算社区欺诈比例
    for comm, nodes in communities.items():
        fraud_count = sum(1 for node in nodes if G.nodes[node].get('label') == 1)
        community_fraud_ratio[comm] = fraud_count / len(nodes) if len(nodes) > 0 else 0

    # 标记高风险社区（欺诈比例 > 30%）
    high_risk_communities = {comm for comm, ratio in community_fraud_ratio.items() if ratio > 0.3}

    # 添加高风险社区标记
    for node in G.nodes():
        comm = G.nodes[node].get('community', -1)
        G.nodes[node]['high_risk_community'] = 1 if comm in high_risk_communities else 0

    # 社区分析
    community_stats = []
    for comm, nodes in communities.items():
        subgraph = G.subgraph(nodes)
        community_stats.append({
            'community': comm,
            'size': len(nodes),
            'fraud_ratio': community_fraud_ratio[comm],
            'high_risk': 1 if comm in high_risk_communities else 0
        })

    df_community = pd.DataFrame(community_stats)
    return G, df_community.sort_values('fraud_ratio', ascending=False)


# 4. 特征提取与筛选
def extract_features(G, df_metrics):
    """提取图特征和基础特征"""
    features = []
    for node in G.nodes():
        # 图特征
        neighbors = list(G.neighbors(node))
        in_degree = G.in_degree(node)
        out_degree = G.out_degree(node)
        avg_calls = np.mean([G[node][n]['calls'] for n in neighbors]) if neighbors else 0

        # 高风险联系人数量
        high_risk_contacts = sum(1 for n in neighbors if G.nodes[n].get('label') == 1)

        # 高风险社区标记
        high_risk_community = G.nodes[node].get('high_risk_community', 0)

        # 网络重要性
        pagerank = df_metrics.loc[node, 'pagerank'] if node in df_metrics.index else 0

        fea_dict = {}
        for col in feature_desc_dict.keys():
            fea_dict[col] = G.nodes[node].get(col, 0)
        graph_dict = {
            'node': node,
            'in_degree': in_degree,
            'out_degree': out_degree,
            'pagerank': pagerank,
            'avg_calls': avg_calls,
            'high_risk_contacts': high_risk_contacts,
            'high_risk_community': high_risk_community,
            'label': G.nodes[node].get('label', 0)
        }
        features_dict = dict(ChainMap(fea_dict, graph_dict))
        features.append(features_dict)

    # 构建特征向量
    df_features = pd.DataFrame(features).set_index('node')
    return df_features


# 特征权重分配
def assign_feature_weights(df_features):
    """分配特征权重（基于领域知识）"""
    if 'label' not in df_features.columns:
        raise ValueError("df_features 中未包含标签列 label")

    X = df_features.drop(columns=['label', 'TERM_BRAND'])
    y = df_features['label']

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'seed': 42
    }

    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=6
    )

    importance_dict = dict(zip(
        X.columns,
        model.feature_importance(importance_type='gain')
    ))

    total_gain = sum(importance_dict.values())
    return {k: v / total_gain for k, v in importance_dict.items() if total_gain > 0}


# 特征描述映射
def get_feature_descriptions():
    """获取特征的可解释描述"""
    return {
        'high_risk_contacts': "高风险联系人",
        'high_risk_community': "高风险社区成员",
        'pagerank': "网络中心位置",
        'TOTAL_CALL_CNT_1M': "异常高频通话",
        'CALL_OPS_NBR_CNT_1M': "多号码联系模式",
        'IMSG_MT_SMS_QTY_1M': "异常短信行为",
        'OWE_FEE_1M': "异常欠费记录",
        'APP_CNT': "借贷APP使用激增",
        'in_degree': "被叫集中度",
        'out_degree': "主叫集中度",
        'avg_calls': "平均通话频次"
    }


# 生成证据链条
def generate_evidence_path(G, user_id, top_features):
    """生成欺诈证据链条"""
    evidence_path = []
    node_data = G.nodes[user_id]

    for feature in top_features:
        if feature == 'high_risk_contacts':
            # 查找高风险联系人
            risky_contacts = [n for n in G.neighbors(user_id) if G.nodes[n].get('label') == 1]
            if risky_contacts:
                contact = risky_contacts[0]
                call_count = G[user_id][contact]['calls']
                duration = G[user_id][contact]['duration']
                evidence_path.append(f"{user_id} → 高风险用户[{contact}](通话{call_count}次,时长{duration}秒)")

        elif feature == 'high_risk_community':
            comm_id = node_data.get('community', -1)
            evidence_path.append(f"高风险社区[{comm_id}]成员")

        elif feature == 'APP_CNT':
            app_count = node_data.get('APP_CNT', 0)
            evidence_path.append(f"使用{app_count}个借贷APP")

        elif feature == 'TOTAL_CALL_CNT_1M':
            call_count = node_data.get('TOTAL_CALL_CNT_1M', 0)
            evidence_path.append(f"近1个月{call_count}次通话")

        elif feature == 'CALL_OPS_NBR_CNT_1M':
            contact_count = node_data.get('CALL_OPS_NBR_CNT_1M', 0)
            evidence_path.append(f"近1个月联系{contact_count}个号码")

        elif feature == 'IMSG_MT_SMS_QTY_1M':
            sms_count = node_data.get('IMSG_MT_SMS_QTY_1M', 0)
            evidence_path.append(f"近1个月发送{sms_count}条短信")

        elif feature == 'OWE_FEE_1M':
            owe_amount = node_data.get('OWE_FEE_1M', 0)
            evidence_path.append(f"近1个月欠费{owe_amount}元")

    return evidence_path


# 生成详细欺诈原因分析
def generate_detailed_reasons(G, user_id, top_features):
    """生成详细的欺诈原因分析"""
    reasons = []
    node_data = G.nodes[user_id]

    # 社交网络维度
    if 'high_risk_contacts' in top_features:
        risky_contacts = [n for n in G.neighbors(user_id) if G.nodes[n].get('label') == 1]
        contact_details = []
        for contact in risky_contacts[:3]:  # 最多展示3个高风险联系人
            call_count = G[user_id][contact]['calls']
            duration = G[user_id][contact]['duration']
            contact_details.append(f"{contact}(通话{call_count}次,时长{duration}秒)")

        reasons.append({
            "维度": "社交网络",
            "原因": f"与{len(risky_contacts)}个已知欺诈用户有联系",
            "详情": contact_details,
            "风险权重": "高"
        })

    # 社区维度
    if 'high_risk_community' in top_features:
        comm_id = node_data.get('community', -1)
        comm_nodes = [n for n in G.nodes() if G.nodes[n].get('community') == comm_id]
        comm_size = len(comm_nodes)
        fraud_in_comm = sum(1 for n in comm_nodes if G.nodes[n].get('label') == 1)

        reasons.append({
            "维度": "社交网络",
            "原因": f"属于高风险社区(社区ID:{comm_id})",
            "详情": [
                f"社区规模: {comm_size}人",
                f"社区内已知欺诈用户: {fraud_in_comm}人",
                f"欺诈比例: {fraud_in_comm / comm_size * 100:.1f}%" if comm_size > 0 else "欺诈比例: 0%"
            ],
            "风险权重": "高"
        })

    # 通话行为维度
    if 'TOTAL_CALL_CNT_1M' in top_features:
        total_calls = node_data.get('TOTAL_CALL_CNT_1M', 0)
        outgoing = node_data.get('CALL_CNT', 0)
        incoming = node_data.get('CALLED_CNT', 0)
        unique_contacts = node_data.get('CALL_OPS_NBR_CNT_1M', 0)

        outgoing_ratio = outgoing / total_calls * 100 if total_calls > 0 else 0

        reasons.append({
            "维度": "通话行为",
            "原因": "异常高频通话模式",
            "详情": [
                f"近1月总通话次数: {total_calls}次",
                f"主叫通话占比: {outgoing_ratio:.1f}%",
                f"联系不同号码数: {unique_contacts}个"
            ],
            "风险权重": "中高"
        })

    # 金融活动维度
    if 'OWE_FEE_1M' in top_features:
        owe_fee = node_data.get('OWE_FEE_1M', 0)
        owe_count = node_data.get('OWE_CNT_1M', 0)
        bank_calls = node_data.get('IN_BANK_CNT', 0) + node_data.get('IN_STOCK_BANK_CNT', 0)
        bank_sms = node_data.get('BANK_SMS_CNT', 0)

        reasons.append({
            "维度": "金融活动",
            "原因": "异常金融活动模式",
            "详情": [
                f"近1月欠费金额: {owe_fee}元",
                f"近1月欠费次数: {owe_count}次",
                f"接收银行相关电话: {bank_calls}次",
                f"接收银行短信: {bank_sms}条"
            ],
            "风险权重": "中"
        })

    # APP使用维度
    if 'APP_CNT' in top_features:
        app_count = node_data.get('APP_CNT', 0)
        online_duration = node_data.get('ONLINE_DUR', 0)
        query_count = node_data.get('QUERY_CNT', 0)

        reasons.append({
            "维度": "APP使用",
            "原因": "异常APP使用行为",
            "详情": [
                f"使用APP数量: {app_count}个",
                f"日均使用时长: {online_duration / 60:.1f}分钟",
                f"日均使用次数: {query_count}次"
            ],
            "风险权重": "中低"
        })

    return reasons


# 生成防御建议
def generate_defense_suggestions(reasons):
    """根据风险原因生成防御建议"""
    suggestions = []

    for reason in reasons:
        dim = reason["维度"]
        if dim == "社交网络":
            suggestions.append("建议措施: 限制该用户与高风险联系人的通信，监控其社交网络变化")
        elif dim == "通话行为":
            suggestions.append("建议措施: 监控异常通话模式，设置通话频率阈值告警")
        elif dim == "金融活动":
            suggestions.append("建议措施: 核查欠费原因，监控银行相关活动")
        elif dim == "APP使用":
            suggestions.append("建议措施: 核查APP使用情况，特别是金融类APP的使用模式")

    # 添加通用建议
    suggestions.extend([
        "通用建议1: 加强用户身份验证",
        "通用建议2: 设置交易额度限制",
        "通用建议3: 定期审查账户活动"
    ])

    return suggestions[:5]  # 返回最多5条建议


# 5. 生成大模型Prompt
def generate_prompts(df_features, feature_weights, G):
    """生成大模型Prompt格式"""
    prompts = []
    feature_descriptions = get_feature_descriptions()

    for user_id, row in df_features.iterrows():
        # 计算风险评分
        risk_score = row['label']
        contributions = []

        for feature, weight in feature_weights.items():
            # 跳过标签列
            if feature == 'label':
                continue

            # 确保特征存在
            if feature not in row:
                continue

            # 计算特征贡献
            feature_value = row[feature]
            contribution = feature_value * weight

            # 存储贡献值
            contributions.append({
                'feature': feature,
                'contribution': contribution
            })


        # 按贡献度排序
        contributions.sort(key=lambda x: x['contribution'], reverse=True)

        # 获取贡献最高的三个特征
        top_contributions = contributions[:3]

        # 获取贡献最高的特征名称
        top_features = [c['feature'] for c in top_contributions]

        # 生成详细原因分析
        detailed_reasons = generate_detailed_reasons(G, user_id, top_features)

        # 生成防御建议
        defense_suggestions = generate_defense_suggestions(detailed_reasons)

        # 确定风险等级
        if risk_score > 0.8:
            risk_level = "极高风险"
            risk_description = "极有可能正在进行欺诈活动，建议立即采取措施"
        elif risk_score > 0.6:
            risk_level = "高风险"
            risk_description = "高度疑似欺诈行为，需要优先调查"
        elif risk_score > 0.4:
            risk_level = "中高风险"
            risk_description = "存在多个可疑特征，建议深入审查"
        elif risk_score > 0.2:
            risk_level = "中风险"
            risk_description = "部分行为特征异常，建议关注"
        else:
            risk_level = "低风险"
            risk_description = "当前未发现明显欺诈特征"

        # 构建输入特征字典（排除标签）
        input_features = {k: v for k, v in row.items() if k != 'label'}

        # 生成证据链
        evidence_path = generate_evidence_path(G, user_id, top_features)

        # 构建Prompt
        prompt = {
            "instruction": "基于知识图谱和用户行为特征的欺诈风险评估",
            "input": {
                "user_id": user_id,
                "features": input_features
            },
            "output": {
                "risk_assessment": {
                    "risk_score": round(risk_score, 4),
                    "risk_level": risk_level,
                    "risk_description": risk_description
                },
                "reason_analysis": detailed_reasons,
                "key_evidence": evidence_path,
                "defense_suggestions": defense_suggestions,
                "contributing_factors": [
                    {
                        "factor": f['feature'],
                        "contribution": round(f['contribution'], 4),
                        "normalized_value": round(row[f['feature']], 4)
                    }
                    for f in top_contributions
            ]
        }
        }

        prompts.append(prompt)

    return prompts

def extract_user_subgraph(graph, user_id, k=2):
    neighbors = set([user_id])
    for _ in k:
        new = set()
        for node in neighbors:
            new.update(graph.neighbors(node))
        neighbors += new
    subgraph = graph.subgraph(neighbors)
    return subgraph

def build_pyg_data_from_nx(subgraph):
    node_map = {n: i for i, n in enumerate(subgraph.nodes())}
    edge_index = torch.tensor([[node_map[u], node_map[v]] for u, v in subgraph.edges()], dtype=torch.long).T
    x = torch.eye(len(node_map))  # One-hot encoding
    return Data(x=x, edge_index=edge_index), node_map














# 主函数
def main(data_path='data.txt'):
    # 1. 构建图谱
    G, data = build_graph(data_path)
    print(f"图谱构建完成，包含 {len(G.nodes())} 个节点和 {len(G.edges())} 条边")

    # 2. 重要节点识别
    df_metrics = identify_key_nodes(G)
    print("\nTop 10 重要节点:")
    print(df_metrics.head(10))

    # 3. 社区发现
    G, df_community = community_detection(G)
    high_risk_comms = df_community[df_community['high_risk'] == 1]
    print(f"\n发现 {len(high_risk_comms)} 个高风险社区:")
    print(high_risk_comms.head())

    # 4. 特征提取
    df_features = extract_features(G, df_metrics)
    print(f"\n提取 {len(df_features)} 个用户的特征")

    # 5. 分配特征权重
    feature_weights = assign_feature_weights(df_features)
    print("\n特征权重分配:")
    for feature, weight in feature_weights.items():
        print(f"{feature}: {weight:.4f}")

    # 6. 生成Prompt
    prompts = generate_prompts(df_features, feature_weights, G)

    # 7. 保存结果
    with open('prompts.jsonl', 'w', encoding='utf-8') as f:
        for prompt in prompts:
            f.write(json.dumps(prompt, ensure_ascii=False) + '\n')

    print(f"\n已生成 {len(prompts)} 条Prompt记录，保存到 prompts.jsonl")

    # 返回前3个prompt示例
    return prompts[:3]


if __name__ == "__main__":
    data_path = 'graph_synthetic_data.txt'
    sample_prompts = main(data_path)
    print("\nPrompt示例:")
    for i, prompt in enumerate(sample_prompts):
        print(f"\n示例 {i + 1}:")
        print(json.dumps(prompt, indent=2, ensure_ascii=False))