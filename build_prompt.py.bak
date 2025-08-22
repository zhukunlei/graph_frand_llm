import networkx as nx
import pandas as pd
import numpy as np
import json
from community import community_louvain
from sklearn.preprocessing import MinMaxScaler
import warnings
from collections import ChainMap  # 确保这行存在
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch.nn import functional as F
import  pickle

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
 #特征到类别的映射（用于选择解释模板）
FEATURE_CATEGORIES = {
    'social': {'high_risk_contacts', 'high_risk_community', 'in_degree', 'out_degree', 'pagerank', 'avg_calls'},
    'call': {'TOTAL_CALL_CNT_1M', 'CALL_CNT', 'CALLED_CNT', 'CALL_OPS_NBR_CNT_1M', 'CALL_CNT_3M_MEAN', 'CALL_CNT_6M_MEAN'},
    'finance': {'OWE_FEE_1M', 'OWE_CNT_1M', 'PAY_RT_1M', 'TOT_FEE_1M'},
    'app': {'APP_CNT', 'ONLINE_DUR', 'QUERY_CNT', 'QUERY_DAYS'},
    'sms': {'IMSG_MT_SMS_QTY_1M', 'BANK_SMS_CNT', 'STOCK_BANK_SMS_CNT', 'PL_BANK_SMS_CNT'},
    # 若你的项目中还有其他类别，按需补充
}

FEATURE_CATEGORIES = {
    'social': {'high_risk_contacts', 'high_risk_community', 'in_degree', 'out_degree', 'pagerank', 'avg_calls'},
    'call':{ 'CALL_CNT', 'CALLED_CNT','TOTAL_CNT', 'CALL_TIME_LENGTH', 'CALLED_TIME_LENGTH','TOTAL_TIME_LENGTH', 'TOTAL_CALL_CNT_3M_SUM', 'TOTAL_CALL_CNT_6M_SUM','TOTAL_CALL_CNT_3M_MEAN', 'TOTAL_CALL_CNT_6M_MEAN','CALLED_CNT_1M', 'CALLED_CNT_3M_SUM','CALLED_CNT_6M_SUM', 'CALL_CNT_3M_MEAN','CALL_CNT_6M_MEAN', 'CALL_OPS_NBR_CNT_1M','CALL_OPS_NBR_CNT_3M_SUM','CALL_OPS_NBR_CNT_6M_SUM','TOTAL_CALL_CNT_1M'},
    'app':{'ONLINE_DUR', 'QUERY_CNT','QUERY_DAYS', 'APP_CNT'} ,
    'user_feature':{'GENDER', 'AGE','CUST_STAR_LVL', 'IS_ENT', 'INNET_MONTHS','PRODUCT_TYPE'},
    'device':{'TERM_TYPE', 'TERM_BRAND'},
    'finance':{'IMSG_MT_SMS_QTY_1M','IMSG_MT_SMS_QTY_3M_SUM', 'IMSG_MT_SMS_QTY_6M_SUM','IN_BANK_CNT', 'IN_STOCK_BANK_CNT','IN_PL_BANK_CNT', 'IN_CONS_FIN_CNT','BANK_SMS_ACCT_CNT','STOCK_BANK_SMS_SCCT_CNT','PL_BANK_SMS_ACCT_CNT', 'BANK_SMS_CNT','STOCK_BANK_SMS_CNT', 'PL_BANK_SMS_CNT','CONS_FIN_SMS_CNT'},
    'consume':{'THIS_ACCT_FEE_TAX', 'PAY_RT_1M','PAY_RT_3M_SUM', 'PAY_RT_6M_SUM','TOT_TAOCAN_FEE_1M', 'TOT_TAOCAN_FEE_3M_SUM','TOT_TAOCAN_FEE_6M_SUM', 'OWE_FEE_1M','OWE_FEE_3M_SUM', 'OWE_FEE_6M_SUM','TOT_FEE_1M', 'TOT_FEE_3M_SUM','TOT_FEE_6M_SUM', 'OWE_CNT_1M','OWE_CNT_3M', 'OWE_CNT_6M'}
}


def get_feature_category(feature):
    for cat, s in FEATURE_CATEGORIES.items():
        if feature in s:
            return cat
    return None

def safe_get_node_attr(G, node, key, default=0):
    if not G.has_node(node):
        return default
    return G.nodes[node].get(key, default)

def safe_edge_info(G, u, v):
    """返回边属性的安全字典（支持无边或有向/无向情况）"""
    if G.has_edge(u, v):
        return G.get_edge_data(u, v) or {}
    # 尝试反方向（如果数据可能是反向存储）
    if G.has_edge(v, u):
        return G.get_edge_data(v, u) or {}
    return {}

def get_feature_descriptions():
    """获取特征的可解释描述"""
    return {
        'high_risk_contacts': "高风险联系人",
        'high_risk_community': "高风险社区成员",
        'pagerank': "社交网络中心位置",
        'TOTAL_CALL_CNT_1M': "异常高频通话",
        'CALL_OPS_NBR_CNT_1M': "多号码联系模式",
        'IMSG_MT_SMS_QTY_1M': "异常短信行为",
        'OWE_FEE_1M': "异常欠费记录",
        'APP_CNT': "借贷APP使用激增",
        'in_degree': "被叫集中度",
        'out_degree': "主叫集中度",
        'avg_calls': "平均通话频次"
    }
# ---------- 替换：generate_evidence_path（现在接收 top_contributions 列表） ----------
def generate_evidence_path(G, user_id, top_contributions):
    """
    生成欺诈证据链条（更健壮）。
    top_contributions: 列表，元素为 {'feature': str, 'contribution': float}
    """
    evidence_path = []
    if not G.has_node(user_id):
        return ["用户在图谱中不存在"]

    node_data = G.nodes[user_id]
    desc_map = dict(ChainMap(cols_dict, feature_desc_dict, get_feature_descriptions()))

    for item in top_contributions:
        feature = item.get('feature')
        contribution = item.get('contribution', None)
        feature_desc = desc_map.get(feature, feature)
        cat = get_feature_category(feature)

        # 社交类（优先尝试生成路径/边信息）
        if cat == 'social':
            if feature == 'high_risk_contacts':
                risky_contacts = [n for n in G.neighbors(user_id) if G.nodes[n].get('label') == 1]
                if risky_contacts:
                    contact = risky_contacts[0]
                    edge = safe_edge_info(G, user_id, contact)
                    call_count = edge.get('calls', edge.get('weight', '未知'))
                    evidence_path.append(f"与高风险用户[{contact}]有联系（通话{call_count}次）")
                else:
                    evidence_path.append("未发现高风险联系人")
            elif feature == 'high_risk_community':
                comm_id = node_data.get('community', -1)
                evidence_path.append(f"所属高风险社区: {comm_id}")
            else:
                # 通用社交特征说明
                val = node_data.get(feature, '未知')
                evidence_path.append(f"用户的社交关系特征具有欺诈嫌疑")

        # 通话类
        elif cat == 'call':
            val = node_data.get(feature, 0)
            evidence_path.append("用户通话频次异常")

        # 金融类
        elif cat == 'finance':
            val = node_data.get(feature, 0)
            evidence_path.append("用户与金融机构联系紧密")

        # APP
        elif cat == 'app':
            val = node_data.get(feature, 0)
            evidence_path.append(f"用户高频次使用借贷类app")

        elif cat == 'device':
            evidence_path.append(f"用户使用可疑欺诈设备")

        elif cat == 'consume':
            evidence_path.append(f"用户的消费行为或金额异常")

        elif cat == 'user_feature':
            evidence_path.append(f"用户的基础属性具有欺诈特性")



        # 未分类：尽量给出值和贡献，避免空
        else:
            val = node_data.get(feature, None)
            if val is not None:
                evidence_path.append(f"{feature_desc}: {val} (贡献: {round(contribution,4) if contribution else 'N/A'})")
            else:
                evidence_path.append(f"{feature_desc}: 无节点值 (贡献: {round(contribution,4) if contribution else 'N/A'})")

    return evidence_path


# ---------- 替换：generate_detailed_reasons（现在接收 top_contributions 列表） ----------

def call_cnt_reason(base_entry, node_data, total_call_feature, call_ops_nbr_featuer, val,  total_calls_threshold=20, k=1):
    total_calls = node_data.get(total_call_feature, node_data.get('TOTAL_CNT', val or 0))
    outgoing = node_data.get('CALL_CNT', 0)
    unique_contacts = node_data.get(call_ops_nbr_featuer, 0)
    outgoing_ratio = (outgoing / total_calls * 100) if total_calls and total_calls > 0 else 0
    base_entry["原因"] = "异常通话频率/结构"
    reason_call_str ="近" + str(k) + "月总通话次数"
    reason_call_rate = "近" + str(k) + "主叫通话占比"
    base_entry["详情"] = [
        reason_call_str + ": {total_calls}",
        reason_call_rate + ": {outgoing_ratio:.1f}%",
        f"联系不同号码数: {unique_contacts}"
    ]
    base_entry["风险权重"] = "中高" if total_calls > total_calls_threshold else "中"
    return base_entry

def generate_detailed_reasons(G, user_id, top_contributions):
    """
    生成详细的欺诈原因分析（更通用）。
    top_contributions: [{'feature':str,'contribution':float}, ...]（通常为前N个贡献最大特征）
    返回：reasons 列表（同原结构）
    """
    reasons = []
    if not G.has_node(user_id):
        return [{"维度": "元数据", "原因": "用户在图谱中不存在", "详情": [], "风险权重": "低"}]

    node_data = G.nodes[user_id]
    desc_map = dict(ChainMap(cols_dict, feature_desc_dict, get_feature_descriptions()))
    user_label = G.nodes[user_id]['label']
    if user_label == '0':
        return [{"维度": "元数据", "原因": "", "详情": [], "风险权重": "低"}]

    # 遍历每个 top contribution，为其生成一条或补充到维度中
    for item in top_contributions:
        feature = item.get('feature')
        contribution = item.get('contribution', 0)
        feature_desc = desc_map.get(feature, feature)
        cat = get_feature_category(feature)
        val = node_data.get(feature, None)

        # 构建基准条目
        base_entry = {
            "维度": None,
            "原因": None,
            "详情": [],
            "风险权重": None
        }

        # 社交网络类
        if cat == 'social':
            base_entry["维度"] = "社交网络"
            if feature == 'high_risk_contacts':
                risky_contacts = [n for n in G.neighbors(user_id) if G.nodes[n].get('label') == 1]
                base_entry["原因"] = f"与{len(risky_contacts)}个已知欺诈用户有联系"
                details = []
                for contact in risky_contacts[:3]:
                    edge = safe_edge_info(G, user_id, contact)
                    call_count = edge.get('calls', '未知')
                    duration = edge.get('duration', '未知')
                    details.append(f"{contact}(通话{call_count}次,时长{duration}秒)")
                base_entry["详情"] = details or ["无详细高风险联系人信息"]
                base_entry["风险权重"] = "高" if len(risky_contacts) > 0 else "低"
            elif feature == 'high_risk_community':
                comm_id = node_data.get('community', -1)
                comm_nodes = [n for n in G.nodes() if G.nodes[n].get('community') == comm_id]
                fraud_in_comm = sum(1 for n in comm_nodes if G.nodes[n].get('label') == 1)
                ratio = (fraud_in_comm / len(comm_nodes) * 100) if len(comm_nodes) > 0 else 0
                base_entry["原因"] = f"所属社区存在较高欺诈率（社区ID:{comm_id}）"
                base_entry["详情"] = [
                    f"社区规模: {len(comm_nodes)}",
                    f"社区内已知欺诈用户: {fraud_in_comm}",
                    f"欺诈比例: {ratio:.1f}%"
                ]
                base_entry["风险权重"] = "高" if ratio > 30 else "中"
            else:
                base_entry["原因"] = f"{feature_desc} 异常"
                base_entry["详情"] = [f"{feature_desc}: {val}", f"模型贡献: {round(contribution,4)}"]
                base_entry["风险权重"] = "中"

        # 通话行为类
        elif cat == 'call':
            base_entry["维度"] = "通话行为"

            if feature in ('TOTAL_CALL_CNT_1M', 'CALL_CNT'):
                """
                total_calls = node_data.get('TOTAL_CALL_CNT_1M', node_data.get('TOTAL_CNT', val or 0))
                outgoing = node_data.get('CALL_CNT', 0)
                unique_contacts = node_data.get('CALL_OPS_NBR_CNT_1M', 0)
                outgoing_ratio = (outgoing / total_calls * 100) if total_calls and total_calls > 0 else 0
                base_entry["原因"] = "异常通话频率/结构"
                base_entry["详情"] = [
                    f"近1月总通话次数: {total_calls}",
                    f"主叫通话占比: {outgoing_ratio:.1f}%",
                    f"联系不同号码数: {unique_contacts}"
                ]
                base_entry["风险权重"] = "中高" if total_calls > 100 else "中"
                """

                base_entry = call_cnt_reason(base_entry, node_data, 'TOTAL_CALL_CNT_1M', 'CALL_OPS_NBR_CNT_1M', val,
                                total_calls_threshold=20, k=1)
            elif feature in ('TOTAL_CALL_CNT_3M', 'CALL_CNT'):
                base_entry = call_cnt_reason(base_entry, node_data, 'TOTAL_CALL_CNT_3M', 'CALL_OPS_NBR_CNT_3M', val,
                                             total_calls_threshold=20, k=3)
            elif feature in ('TOTAL_CALL_CNT_6M', 'CALL_CNT'):
                base_entry = call_cnt_reason(base_entry, node_data, 'TOTAL_CALL_CNT_6M', 'CALL_OPS_NBR_CNT_6M', val,
                                             total_calls_threshold=20, k=6)
            else:
                total_calls = node_data.get('TOTAL_CALL_CNT_1M', node_data.get('TOTAL_CNT', val or 0))
                outgoing = node_data.get('CALL_CNT', 0)
                unique_contacts = node_data.get('CALL_OPS_NBR_CNT_1M', 0)
                outgoing_ratio = (outgoing / total_calls * 100) if total_calls and total_calls > 0 else 0
                base_entry["原因"] = "异常通话频率/结构"
                base_entry["详情"] = [
                    f"总通话次数: {total_calls}",
                    f"主叫通话占比: {outgoing_ratio:.1f}%",
                    f"联系不同号码数: {unique_contacts}"
                ]
                base_entry["风险权重"] = "中高" if total_calls > 20 else "中"

        elif cat == 'device':
            base_entry["原因"] = '使用高欺诈嫌疑的设备'
            base_entry["详情"] = [f"类似设备被欺诈用户高频使用"]
            base_entry["风险权重"] = "低"

        # 金融类
        elif cat == 'finance':
            owe_fee = node_data.get('OWE_FEE_1M', node_data.get(feature, 0))
            owe_count = node_data.get('OWE_CNT_1M', 0)

            bank_calls = node_data.get('IN_BANK_CNT', 0) + node_data.get('IN_STOCK_BANK_CNT', 0) + node_data.get('IN_PL_BANK_CNT', 0)
            comsume_finance_calls = node_data.get('IN_COMSUME_FINANCE_CNT', 0)

            bank_sms_cnt = node_data.get('BANK_SMS_ACCT_CNT', 0) + node_data.get('STOCK_BANK_SMS_SCCT_CNT', 0) + node_data.get('PL_BANK_SMS_ACCT_CNT', 0) + node_data.get('BANK_SMS_CNT', 0) + node_data.get('STOCK_BANK_SMS_CNT', 0) + node_data.get('PL_BANK_SMS_CNT', 0)
            consume_finance_sms = node_data.get('CONS_FIN_SMS_CNT', 0)

            base_entry["维度"] = "金融活动"
            base_entry["原因"] = "异常金融行为"
            base_entry["详情"] = [
                f"接收银行相关电话: {bank_calls}次",
                f"接收消费金融公司相关电话： {comsume_finance_calls}次",
                f"接收银行相关的短信: {bank_sms_cnt}次",
                f"接收消费金融公司的相关短信：{consume_finance_sms}次"
            ]
            base_entry["风险权重"] = "中高" if owe_fee and owe_fee > 100 else "中"

        # APP / 使用类
        elif cat == 'app':
            app_count = node_data.get('APP_CNT', node_data.get(feature, 0))
            online_duration = node_data.get('ONLINE_DUR', 0)
            query_count = node_data.get('QUERY_CNT', 0)
            base_entry["维度"] = "APP使用"
            base_entry["原因"] = "异常APP或线上行为"
            base_entry["详情"] = [
                f"使用APP数量: {app_count}",
                f"总在线时长: {online_duration}秒",
                f"使用次数: {query_count}"
            ]
            base_entry["风险权重"] = "中" if app_count and app_count > 5 else "中低"

        # 短信 / 通知类
        elif cat == 'consume':

            if feature in ('OWE_FEE_1M','OWE_FEE_3M_SUM', 'OWE_FEE_6M_SUM','TOT_FEE_1M', 'TOT_FEE_3M_SUM','TOT_FEE_6M_SUM', 'OWE_CNT_1M','OWE_CNT_3M', 'OWE_CNT_6M'):
                owe_fee = node_data.get('TOT_FEE_6M_SUM', node_data.get(feature, 0))
                owe_count = node_data.get('OWE_CNT_6M', node_data.get(feature, 0))

                base_entry["维度"] = "用户通信欠费"
                base_entry["原因"] = f"消费情况异常"
                base_entry["详情"] = [
                    f"用户通信欠费金额：{owe_fee}元",
                    f"用户通信欠费次数：{owe_count}次"
                ]
                base_entry["风险权重"] = "中"
            else:
                pay_fee = node_data.get('TOT_TAOCAN_FEE_6M_SUM', node_data.get(feature, 0))
                base_entry["维度"] = "用户通信消费"
                base_entry["原因"] = f"消费消费异常"
                base_entry["详情"] = [
                    f"用户通信消费金额：{pay_fee}元"
                ]
                base_entry["风险权重"] = "低"


        # 未分类（回退）
        else:
            base_entry["维度"] = "其它"
            base_entry["原因"] = f"{feature_desc} 影响模型判断"
            base_entry["详情"] = [f"{feature_desc}: {val}", f"模型贡献: {round(contribution,4)}"]
            base_entry["风险权重"] = "中"

        reasons.append(base_entry)

    # 若没有生成任何理由（极少出现），则构造一个回退说明，避免空返回
    if not reasons:
        # 构造基于 top_contributions 的摘要
        fallback_details = []
        for it in top_contributions:
            f = it.get('feature')
            c = it.get('contribution', 0)
            v = node_data.get(f, '无')
            fallback_details.append(f"{desc_map.get(f,f)}={v} (贡献={round(c,4)})")
        reasons.append({
            "维度": "模型贡献摘要",
            "原因": "模型基于下列特征判断为可疑",
            "详情": fallback_details,
            "风险权重": "中"
        })

    return reasons


# ---------- 替换：generate_prompts 中调用部分（切换为传递 top_contributions 而非 top_features） ----------
#def generate_prompts(df_features, feature_weights, G):
def remove_duplicates_dict_list(lst):
    seen = set()
    result = []
    for d in lst:
        # 将字典转换为可哈希的元组
        dict_tuple = tuple(sorted(d.items()))
        if dict_tuple not in seen:
            seen.add(dict_tuple)
            result.append(d)
    return result


def generate_prompts(features_file="features.csv",
                         feature_weights_file="feature_weights.json",
                         graph_file="community_graph.gpickle",
                         evidence_file="evidence.json",
                         reasons_file="detailed_reasons.json",
                         output_file="prompts.json",
                            k=3):
    """生成大模型Prompt格式（已修改：传递 top_contributions）"""

    add_desc_dict = {
        'high_risk_contacts': "高风险联系人",
        'high_risk_community': "高风险社区成员",
        'pagerank': "社交网络中心位置",
        'in_degree': "被叫集中度",
        'out_degree': "主叫集中度",
        'avg_calls': "平均通话频次"
    }
    cols_dict.update(add_desc_dict)

    df_features = pd.read_csv(features_file, index_col=0)
    with open(feature_weights_file, "r", encoding="utf-8") as f:
        feature_weights = json.load(f)
    with open(graph_file, "rb") as f:
        G = pickle.load(f)
    prompts = []
    feature_descriptions = get_feature_descriptions()
    reason_dict = {}
    evidence_dict = {}

    for user_id, row in df_features.iterrows():
        # 计算风险评分
        risk_score = row['label']
        contributions = []

        for feature, weight in feature_weights.items():
            # 跳过标签列
            if feature == 'label':
                continue

            # 确保特征存在于行中
            if feature not in row:
                continue

            feature_desc_str = cols_dict.get(feature)
            feature_value = row[feature]
            contribution = feature_value * weight
            contributions.append({
                'feature': feature_desc_str,
                'contribution': contribution
            })

        # 按贡献度排序
        contributions.sort(key=lambda x: x['contribution'], reverse=True)

        # 获取贡献最高的三个特征（带贡献值）
        top_contributions = contributions[:k]

        # 生成详细原因分析（现在接收 top_contributions）
        detailed_reasons = generate_detailed_reasons(G, user_id, top_contributions)
        detailed_reasons = remove_duplicates_dict_list(detailed_reasons)
        reason_dict[user_id] = detailed_reasons
        # 生成防御建议
        #defense_suggestions = generate_defense_suggestions(detailed_reasons)

        # 生成证据链（现在接收 top_contributions）
        evidence_path = generate_evidence_path(G, user_id, top_contributions)
        evidence_path = remove_duplicates_dict_list(evidence_path)
        evidence_dict[user_id] = evidence_path
        # 确定风险等级（保持原逻辑）
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
        input_features = {cols_dict.get(k): v for k, v in row.items() if k != 'label'}

        # 构建contributing_factors字段（保留原有输出格式）
        contributing_factors = [
            {
                "factor": f['feature'],
                "contribution": round(f['contribution'], 4),
                "normalized_value": round(row[f['feature']], 4)
            }
            for f in top_contributions
        ]

        # 构建Prompt对象（与原结构兼容）
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
                "contributing_factors": contributing_factors
            }
        }

        prompts.append(prompt)
    with open(reasons_file,"w", encoding="utf-8") as f:
        f.write(json.dumps(reason_dict, ensure_ascii=False))

    with open(evidence_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(evidence_dict, ensure_ascii=False))

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(prompts, ensure_ascii=False))
    #return prompts, reason_dict, evidence_dict

if __name__ == "__main__":
    generate_prompts(features_file="features.csv",
                     feature_weights_file="feature_weights.json",
                     graph_file="community_graph.gpickle",
                     evidence_file="evidence.json",
                     reasons_file="detailed_reasons.json",
                     output_file="prompts.json",
                     k=3)