import networkx as nx
import pandas as pd
import numpy as np
import json
from community import community_louvain
from sklearn.preprocessing import MinMaxScaler
import warnings
from collections import ChainMap

warnings.filterwarnings('ignore')

cols_dict = {'user_id': '用户标识符', 'date':'日期', 'year': '年份', 'label':'是否是欺诈用户', 'opp_user_id':'通话对端用户', 'CALL_CNT': '通话次数', 'CALLED_CNT': '被叫通话次数', 'TOTAL_CNT': '总通话次数', 'CALL_TIME_LENGTH': '主叫通话时长', 'CALLED_TIME_LENGTH': '被叫通话时长', 'TOTAL_TIME_LENGTH': '总通话时长 ', 'ONLINE_DUR': 'app使用时长', 'QUERY_CNT': '使用次数', 'QUERY_DAYS': '使用天数', 'APP_CNT': '访问app次数', 'GENDER': '性别', 'AGE': '年龄', 'CUST_STAR_LVL': '星级', 'IS_ENT': '是否与集团相关', 'INNET_MONTHS': '在网月份', 'PRODUCT_TYPE': '套餐类型', 'TERM_TYPE': '终端类型', 'TERM_BRAND': '终端品牌', 'THIS_ACCT_FEE_TAX': '当月消费总金额 ', 'TOTAL_CALL_CNT_1M': '近1月总通话次数 ', 'TOTAL_CALL_CNT_3M_SUM': '近3月总通话次数 ', 'TOTAL_CALL_CNT_6M_SUM': '近6月总通话次数 ', 'TOTAL_CALL_CNT_3M_MEAN': '近3月平均通话次数', 'TOTAL_CALL_CNT_6M_MEAN': '近6月平均通话次数', 'CALLED_CNT_1M': '近1月呼入通话次数 ', 'CALLED_CNT_3M_SUM': '近3月呼入通话次数 ', 'CALLED_CNT_6M_SUM': '近6月呼入通话次数 ', 'CALL_CNT_3M_MEAN': '近三个月的平均通话次数', 'CALL_CNT_6M_MEAN': '近六个月的平均通话次数', 'CALL_OPS_NBR_CNT_1M': '近1月不重复通话对端数的加和 ', 'CALL_OPS_NBR_CNT_3M_SUM': '近3月不重复通话对端数的加和', 'CALL_OPS_NBR_CNT_6M_SUM': '近6月不重复通话对端数的加和 ', 'IMSG_MT_SMS_QTY_1M': '近1月行业短信接收量 ', 'IMSG_MT_SMS_QTY_3M_SUM': '近3月行业短信接收量 ', 'IMSG_MT_SMS_QTY_6M_SUM': '近6月行业短信接收量 ', 'IN_BANK_CNT': '当月接听国有大行电话的次数 ', 'IN_STOCK_BANK_CNT': '当月接听股份制银行电话的次数 ', 'IN_PL_BANK_CNT': '当月接听民营银行电话的次数 ', 'IN_CONS_FIN_CNT': '当月接听重点消费金融公司电话的次数 ', 'BANK_SMS_ACCT_CNT': '当月接收国有大行动账短信条数 ', 'STOCK_BANK_SMS_SCCT_CNT': '当月接收股份制银行动账短信条数 ', 'PL_BANK_SMS_ACCT_CNT': '当月接收民营银行动账短信条数 ', 'BANK_SMS_CNT': '当月接收国有大行短信条数 ', 'STOCK_BANK_SMS_CNT': '当月接收股份制银行短信条数 ', 'PL_BANK_SMS_CNT': '当月接收民营银行短信条数 ', 'CONS_FIN_SMS_CNT': '当月接收重点消费金融公司短信条数 ', 'PAY_RT_1M': '近1月缴费金额的加和(元) ', 'PAY_RT_3M_SUM': '近3月缴费金额的加和(元) ', 'PAY_RT_6M_SUM': '近6月缴费金额的加和(元) ', 'TOT_TAOCAN_FEE_1M': '近1月套餐消费总金额(元) ', 'TOT_TAOCAN_FEE_3M_SUM': '近3月套餐消费总金额(元)', 'TOT_TAOCAN_FEE_6M_SUM': '近6月套餐消费总金额(元)', 'OWE_FEE_1M': '近1月欠费总金额(元)', 'OWE_FEE_3M_SUM': '近3月欠费总金额(元) ', 'OWE_FEE_6M_SUM': '近6月欠费总金额(元)', 'TOT_FEE_1M': '近1月消费金额(元) ', 'TOT_FEE_3M_SUM': '近3月消费金额(元) ', 'TOT_FEE_6M_SUM': '近6月消费金额(元) ', 'OWE_CNT_1M': '近1月余额为负次数 ', 'OWE_CNT_3M': '近3月余额为负次数', 'OWE_CNT_6M': '近6月余额为负次数 '}
feature_desc_dict  = { 'ONLINE_DUR': 'app使用时长', 'QUERY_CNT': '使用次数', 'QUERY_DAYS': '使用天数', 'APP_CNT': '访问app次数', 'GENDER': '性别', 'AGE': '年龄', 'CUST_STAR_LVL': '星级', 'IS_ENT': '是否与集团相关', 'INNET_MONTHS': '在网月份', 'PRODUCT_TYPE': '套餐类型', 'TERM_TYPE': '终端类型', 'TERM_BRAND': '终端品牌', 'THIS_ACCT_FEE_TAX': '当月消费总金额 ', 'TOTAL_CALL_CNT_1M': '近1月总通话次数 ', 'TOTAL_CALL_CNT_3M_SUM': '近3月总通话次数 ', 'TOTAL_CALL_CNT_6M_SUM': '近6月总通话次数 ', 'TOTAL_CALL_CNT_3M_MEAN': '近3月平均通话次数', 'TOTAL_CALL_CNT_6M_MEAN': '近6月平均通话次数', 'CALLED_CNT_1M': '近1月呼入通话次数 ', 'CALLED_CNT_3M_SUM': '近3月呼入通话次数 ', 'CALLED_CNT_6M_SUM': '近6月呼入通话次数 ', 'CALL_CNT_3M_MEAN': '近三个月的平均通话次数', 'CALL_CNT_6M_MEAN': '近六个月的平均通话次数', 'CALL_OPS_NBR_CNT_1M': '近1月不重复通话对端数的加和 ', 'CALL_OPS_NBR_CNT_3M_SUM': '近3月不重复通话对端数的加和', 'CALL_OPS_NBR_CNT_6M_SUM': '近6月不重复通话对端数的加和 ', 'IMSG_MT_SMS_QTY_1M': '近1月行业短信接收量 ', 'IMSG_MT_SMS_QTY_3M_SUM': '近3月行业短信接收量 ', 'IMSG_MT_SMS_QTY_6M_SUM': '近6月行业短信接收量 ', 'IN_BANK_CNT': '当月接听国有大行电话的次数 ', 'IN_STOCK_BANK_CNT': '当月接听股份制银行电话的次数 ', 'IN_PL_BANK_CNT': '当月接听民营银行电话的次数 ', 'IN_CONS_FIN_CNT': '当月接听重点消费金融公司电话的次数 ', 'BANK_SMS_ACCT_CNT': '当月接收国有大行动账短信条数 ', 'STOCK_BANK_SMS_SCCT_CNT': '当月接收股份制银行动账短信条数 ', 'PL_BANK_SMS_ACCT_CNT': '当月接收民营银行动账短信条数 ', 'BANK_SMS_CNT': '当月接收国有大行短信条数 ', 'STOCK_BANK_SMS_CNT': '当月接收股份制银行短信条数 ', 'PL_BANK_SMS_CNT': '当月接收民营银行短信条数 ', 'CONS_FIN_SMS_CNT': '当月接收重点消费金融公司短信条数 ', 'PAY_RT_1M': '近1月缴费金额的加和(元) ', 'PAY_RT_3M_SUM': '近3月缴费金额的加和(元) ', 'PAY_RT_6M_SUM': '近6月缴费金额的加和(元) ', 'TOT_TAOCAN_FEE_1M': '近1月套餐消费总金额(元) ', 'TOT_TAOCAN_FEE_3M_SUM': '近3月套餐消费总金额(元)', 'TOT_TAOCAN_FEE_6M_SUM': '近6月套餐消费总金额(元)', 'OWE_FEE_1M': '近1月欠费总金额(元)', 'OWE_FEE_3M_SUM': '近3月欠费总金额(元) ', 'OWE_FEE_6M_SUM': '近6月欠费总金额(元)', 'TOT_FEE_1M': '近1月消费金额(元) ', 'TOT_FEE_3M_SUM': '近3月消费金额(元) ', 'TOT_FEE_6M_SUM': '近6月消费金额(元) ', 'OWE_CNT_1M': '近1月余额为负次数 ', 'OWE_CNT_3M': '近3月余额为负次数', 'OWE_CNT_6M': '近6月余额为负次数'}



# 1. 数据准备与图谱构建
def build_graph(data_path):
    """构建通话社交网络图"""
    #s = 'user_id|date|year|label|opp_user_id|CALL_CNT|CALLED_CNT|TOTAL_CNT|CALL_TIME_LENGTH|CALLED_TIME_LENGTH|TOTAL_TIME_LENGTH|ONLINE_DUR|QUERY_CNT|QUERY_DAYS|APP_CNT|GENDER|AGE|CUST_STAR_LVL|IS_ENT|INNET_MONTHS|PRODUCT_TYPE|TERM_TYPE|TERM_BRAND|THIS_ACCT_FEE_TAX|TOTAL_CALL_CNT_1M|TOTAL_CALL_CNT_3M_SUM|TOTAL_CALL_CNT_6M_SUM|TOTAL_CALL_CNT_3M_MEAN|TOTAL_CALL_CNT_6M_MEAN|CALLED_CNT_1M|CALLED_CNT_3M_SUM|CALLED_CNT_6M_SUM|CALL_CNT_3M_MEAN| CALL_CNT_6M_MEAN|CALL_OPS_NBR_CNT_1M|CALL_OPS_NBR_CNT_3M_SUM|CALL_OPS_NBR_CNT_6M_SUM|IMSG_MT_SMS_QTY_1M|IMSG_MT_SMS_QTY_3M_SUM|IMSG_MT_SMS_QTY_6M_SUM|IN_BANK_CNT|IN_STOCK_BANK_CNT|IN_PL_BANK_CNT|IN_CONS_FIN_CNT|BANK_SMS_ACCT_CNT|STOCK_BANK_SMS_SCCT_CNT|PL_BANK_SMS_ACCT_CNT|BANK_SMS_CNT|STOCK_BANK_SMS_CNT|PL_BANK_SMS_CNT|CONS_FIN_SMS_CNT|PAY_RT_1M|PAY_RT_3M_SUM|PAY_RT_6M_SUM| TOT_TAOCAN_FEE_1M|TOT_TAOCAN_FEE_3M_SUM|TOT_TAOCAN_FEE_6M_SUM|OWE_FEE_1M|OWE_FEE_3M_SUM|OWE_FEE_6M_SUM|TOT_FEE_1M|TOT_FEE_3M_SUM|TOT_FEE_6M_SUM|OWE_CNT_1M|OWE_CNT_3M|OWE_CNT_6M'
    cols = cols_dict.keys()
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
                if key == 'ONLINE_DUR':
                    print('find the online_dur feature')
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
def assign_feature_weights():
    """分配特征权重（基于领域知识）"""
    feature_weights = {
        'high_risk_contacts': 0.35,  # 高风险联系人
        'high_risk_community': 0.25,  # 高风险社区
        'pagerank': 0.15,  # 网络中心性
        'call_cnt_1m': 0.08,  # 高频通话
        'contact_cnt_1m': 0.05,  # 多号码联系
        'sms_cnt_1m': 0.04,  # 异常短信
        'owe_fee_1m': 0.03,  # 欠费记录
        'app_cnt': 0.02,  # 借贷APP数量
        'in_degree': 0.01,  # 被叫集中度
        'out_degree': 0.01,  # 主叫集中度
        'avg_calls': 0.01  # 平均通话频次
    }

    # 归一化权重
    total_weight = sum(feature_weights.values())
    for feature in feature_weights:
        feature_weights[feature] /= total_weight

    return feature_weights


# 特征描述映射
def get_feature_descriptions():
    """获取特征的可解释描述"""
    return {
        'high_risk_contacts': "高风险联系人",
        'high_risk_community': "高风险社区成员",
        'pagerank': "网络中心位置",
        'call_cnt_1m': "异常高频通话",
        'contact_cnt_1m': "多号码联系模式",
        'sms_cnt_1m': "异常短信行为",
        'owe_fee_1m': "异常欠费记录",
        'app_cnt': "借贷APP使用激增",
        'in_degree': "被叫集中度",
        'out_degree': "主叫集中度",
        'avg_calls': "平均通话频次"
    }


# 生成证据链条
def generate_evidence_path(G, user_id, top_features):
    """生成欺诈证据链条"""
    evidence_path = []

    for feature in top_features:
        if feature == 'high_risk_contacts':
            # 查找高风险联系人
            risky_contacts = [n for n in G.neighbors(user_id) if G.nodes[n].get('label') == 1]
            if risky_contacts:
                contact = risky_contacts[0]
                evidence_path.append(f"{user_id}→高风险用户[{contact}]")

        elif feature == 'high_risk_community':
            comm_id = G.nodes[user_id].get('community', -1)
            evidence_path.append(f"高风险社区[{comm_id}]成员")

        elif feature == 'app_cnt':
            app_count = G.nodes[user_id].get('app_cnt', 0)
            evidence_path.append(f"使用{app_count}个借贷APP")

        elif feature == 'call_cnt_1m':
            call_count = G.nodes[user_id].get('TOTAL_CALL_CNT_1M', 0)
            evidence_path.append(f"近1个月{call_count}次通话")

        elif feature == 'contact_cnt_1m':
            contact_count = G.nodes[user_id].get('CALL_OPS_NBR_CNT_1M', 0)
            evidence_path.append(f"近1个月联系{contact_count}个号码")

        elif feature == 'sms_cnt_1m':
            sms_count = G.nodes[user_id].get('IMSG_MT_SMS_QTY_1M', 0)
            evidence_path.append(f"近1个月发送{sms_count}条短信")

        elif feature == 'owe_fee_1m':
            owe_amount = G.nodes[user_id].get('OWE_FEE_1M', 0)
            evidence_path.append(f"近1个月欠费{owe_amount}元")

    return evidence_path


# 5. 生成大模型Prompt
def generate_prompts(df_features, feature_weights, G):
    """生成大模型Prompt格式"""
    prompts = []
    feature_descriptions = get_feature_descriptions()

    for user_id, row in df_features.iterrows():
        # 计算风险评分
        risk_score = 0
        contributions = []

        for feature, weight in feature_weights.items():
            # 跳过标签列
            if feature == 'label':
                continue

            # 计算特征贡献
            feature_value = row[feature]
            contribution = feature_value * weight
            risk_score += contribution

            # 存储贡献值
            contributions.append({
                'feature': feature,
                'contribution': contribution
            })

        # 限制风险评分在0-1之间
        risk_score = min(max(risk_score, 0), 1)

        # 按贡献度排序
        contributions.sort(key=lambda x: x['contribution'], reverse=True)

        # 获取贡献最高的三个特征
        top_contributions = contributions[:3]
        reasons = []

        for contrib in top_contributions:
            feature = contrib['feature']
            reasons.append({
                "factor": feature_descriptions.get(feature, feature),
                "contribution": round(contrib['contribution'], 4)
            })

        # 生成证据链条
        evidence_path = generate_evidence_path(G, user_id, [c['feature'] for c in top_contributions])

        # 构建输入特征字典（排除标签）
        input_features = {k: v for k, v in row.items() if k != 'label'}

        # 构建Prompt
        prompt = {
            "instruction": "根据用户的图特征与其它特征，判断该用户的欺诈风险",
            "input": input_features,
            "output": {
                "risk_score": round(risk_score, 4),
                "decision": "高风险" if risk_score > 0.6 else "中风险" if risk_score > 0.3 else "低风险",
                "reasons": reasons,
                "evidence_path": evidence_path
            }
        }

        prompts.append(prompt)

    return prompts


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
    feature_weights = assign_feature_weights()
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
    data_path = 'data.txt'
    sample_prompts = main(data_path)
    print("\nPrompt示例:")
    for i, prompt in enumerate(sample_prompts):
        print(f"\n示例 {i + 1}:")
        print(json.dumps(prompt, indent=2, ensure_ascii=False))