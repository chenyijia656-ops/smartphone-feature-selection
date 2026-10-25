# -*- coding: utf-8 -*-
"""UCI HAR / GRRF 项目图表与中文数据表生成器。


"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

项目目录 = Path.cwd()
结果目录 = 项目目录 / 'results'
图片目录 = 结果目录 / 'figures' / '中文汇报'
中文表目录 = 结果目录 / '中文汇总'
if not 结果目录.exists():
    raise FileNotFoundError('找不到 results 文件夹：请先在 VS Code 终端切换到项目根目录。')
图片目录.mkdir(parents=True, exist_ok=True)
中文表目录.mkdir(parents=True, exist_ok=True)

# 自动优先选择 Windows 常用中文字体；其他系统尝试使用 Noto CJK 字体。
候选字体 = ['Microsoft YaHei', 'SimHei', 'DengXian', 'SimSun',
          'Noto Sans CJK SC', 'Noto Serif CJK SC', 'WenQuanYi Micro Hei']
已安装字体 = {f.name for f in font_manager.fontManager.ttflist}
字体 = next((name for name in 候选字体 if name in 已安装字体), None)
if 字体:
    plt.rcParams['font.sans-serif'] = [字体, 'DejaVu Sans']
    print('使用中文字体：', 字体)
else:
    print('【字体提示】未找到常见中文字体。若图片中文字显示为方框，请在 Windows 安装/启用微软雅黑。')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 11

图数 = 0
表数 = 0

def 读取表(文件名):
    路径 = 结果目录 / 文件名
    if not 路径.exists():
        print('【跳过】不存在：', 文件名)
        return None
    try:
        表 = pd.read_csv(路径)
        print('【读取】', 文件名, '行列数=', 表.shape)
        return 表
    except Exception as 错误:
        print('【跳过】读取失败：', 文件名, 错误)
        return None


def 寻找列(表, 候选):
    return next((名称 for 名称 in 候选 if 名称 in 表.columns), None)


def 保存图(图, 名称):
    global 图数
    图.tight_layout()
    路径 = 图片目录 / 名称
    图.savefig(路径, dpi=220, bbox_inches='tight')
    plt.close(图)
    图数 += 1
    print('【已生成图片】', 路径.relative_to(项目目录))


def 保存中文表(表, 名称, 列对应):
    global 表数
    表 = 表.rename(columns=列对应).copy()
    路径 = 中文表目录 / 名称
    表.to_csv(路径, index=False, encoding='utf-8-sig')
    表数 += 1
    print('【已生成中文表】', 路径.relative_to(项目目录))


def 方法中文(名称):
    return {
        'Full_561': '全特征（561维）',
        'GRRF_gamma_0': 'GRRF（γ=0）',
        'GRRF_gamma_0.25': 'GRRF（γ=0.25）',
        'GRRF_gamma_0.5': 'GRRF（γ=0.5）',
        'GRRF_gamma_0.75': 'GRRF（γ=0.75）',
        'GRRF_gamma_1': 'GRRF（γ=1）',
        'nt100_m23': '100棵树，mtry=23',
        'nt100_m56': '100棵树，mtry=56',
        'nt500_m23': '500棵树，mtry=23',
        'nt500_m56': '500棵树，mtry=56',
        'Common_6': '共同6个特征',
        'Common_6_plus_gamma075_feature': '共同6个+重力加速度特征',
        'Common_6_plus_gamma1_feature': '共同6个+身体加速度变化频域特征',
        'paper_C0.1_gamma0.5_no_scaler': '论文参数，不额外标准化',
        'paper_C0.1_gamma0.5_with_scaler': '论文参数，额外标准化',
        'project_C1_gamma_scale_with_scaler': '项目基线参数+标准化',
        'C0.1_gamma_scale_with_scaler': 'C=0.1，γ=scale+标准化',
    }.get(str(名称), str(名称))


def 加图例条(图表, 名称列, 正确率列, f1列, 标题, 文件名, 图片宽=11):
    表 = 图表.copy()
    if not all(列 in 表.columns for 列 in [名称列, 正确率列, f1列]):
        print('【跳过】缺少图表所需列：', 标题)
        return
    位置 = np.arange(len(表))
    图, 坐标 = plt.subplots(figsize=(图片宽, 6))
    宽 = .36
    坐标.bar(位置 - 宽/2, 表[正确率列].astype(float) * 100, 宽, label='准确率')
    坐标.bar(位置 + 宽/2, 表[f1列].astype(float) * 100, 宽, label='宏平均F1')
    坐标.set_xticks(位置)
    坐标.set_xticklabels([方法中文(值) for 值 in 表[名称列]], rotation=15, ha='right')
    坐标.set_ylim(0, 105)
    坐标.set_ylabel('得分（%）')
    坐标.set_title(标题)
    坐标.legend()
    坐标.grid(axis='y', alpha=.25)
    保存图(图, 文件名)


def 研究流程图():
    图, 坐标 = plt.subplots(figsize=(11, 8))
    坐标.axis('off')
    文字 = [
        'UCI HAR数据集：官方训练集7352条，561个工程特征',
        '建立完整特征SVM基线：标准化 + RBF核SVM',
        '普通随机森林计算重要性 → GRRF筛选特征',
        '按照参与者进行5折分组验证：每折独立筛选特征',
        '比较准确率、宏平均F1、特征数量、耗时与稳定性',
        '后续：论文10折复现、GRF/ReliefF、MI与可学习门控',
    ]
    高度 = np.linspace(.91, .09, len(文字))
    for i, (y, 内容) in enumerate(zip(高度, 文字)):
        坐标.text(.5, y, 内容, ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle='round,pad=.7', fill=False, linewidth=1.3))
        if i < len(文字)-1:
            坐标.annotate('', xy=(.5, 高度[i+1]+.045), xytext=(.5, y-.045),
                        arrowprops=dict(arrowstyle='->', lw=1.3))
    坐标.set_title('项目研究流程及数据泄漏控制', pad=15)
    保存图(图, '00_项目研究流程.png')


def 参数敏感性图():
    表 = 读取表('grrf_gamma_diagnostic_summary.csv')
    if 表 is None: return
    x = 寻找列(表, ['gamma_grrf', 'gamma'])
    y = 寻找列(表, ['selected_features', 'n_features'])
    if not x or not y: return
    表 = 表.sort_values(x)
    图, 坐标 = plt.subplots()
    坐标.plot(表[x], 表[y], marker='o')
    for _, 行 in 表.iterrows():
        坐标.annotate(str(int(行[y])), (行[x], 行[y]), xytext=(6, 6), textcoords='offset points')
    坐标.set(xlabel='GRRF引导参数γ', ylabel='选中特征数量（个）',
           title='引导参数γ对GRRF特征数量的影响（单次诊断）')
    坐标.grid(alpha=.25)
    保存图(图, '01_参数敏感性.png')
    保存中文表(表, '01_参数敏感性汇总.csv', {
        'gamma_grrf': 'GRRF引导参数γ', 'ntree': '树数量', 'mtry': '每次分裂候选特征数',
        'selected_features': '选中特征数量', 'feature_reduction_percent': '特征压缩比例（%）'
    })


def 五折核心对照():
    表 = 读取表('grrf_cv5_summary.csv')
    if 表 is None: return
    名称 = 寻找列(表, ['method', 'config_id'])
    if not 名称:return
    表 = 表[表[名称].isin(['Full_561', 'GRRF_gamma_0.25', 'GRRF_gamma_1'])].copy()
    if 表.empty:return
    加图例条(表, 名称, 'accuracy_mean', 'macro_f1_mean',
         '5折按用户分组验证：分类性能比较（C=1，γ=scale）', '02_五折核心分类性能.png')
    表[名称] = 表[名称].map(方法中文)
    保存中文表(表, '02_五折核心汇总.csv', {
        名称:'方法', 'n_features_mean':'平均特征数', 'n_features_std':'特征数标准差',
        'accuracy_mean':'准确率均值', 'accuracy_std':'准确率标准差',
        'macro_f1_mean':'宏平均F1均值', 'macro_f1_std':'宏平均F1标准差',
        'training_seconds_mean':'模型训练时间均值（秒）',
        'prediction_seconds_mean':'预测时间均值（秒）'})


def 五折逐折图():
    表 = 读取表('grrf_cv5_fold_metrics.csv')
    if 表 is None:return
    折列, 名称, 指标 = 寻找列(表, ['fold', 'validation_fold']), 寻找列(表, ['method', 'config_id']), 'accuracy'
    if not 折列 or not 名称 or 指标 not in 表.columns:return
    图, 坐标 = plt.subplots()
    for 模型, 子表 in 表.groupby(名称):
        子表=子表.sort_values(折列)
        坐标.plot(子表[折列], 子表[指标]*100, marker='o', label=方法中文(模型))
    坐标.set(xlabel='验证折序号', ylabel='准确率（%）', title='5折逐折准确率：不同参与者划分的影响')
    坐标.set_xticks(sorted(表[折列].unique()))
    坐标.set_ylim(0, 100)
    坐标.legend(fontsize=9)
    坐标.grid(alpha=.25)
    保存图(图, '03_五折逐折准确率.png')


def 论文参数网格图():
    表 = 读取表('grrf_gamma1_grid_fair_svm_summary.csv')
    if 表 is None:return
    名称 = 寻找列(表, ['config_id'])
    if not 名称:return
    加图例条(表, 名称, 'accuracy_mean', 'macro_f1_mean',
        '论文γ=1下四组参数的公平比较（统一SVM参数）', '04_gamma1四组公平比较.png', 12)
    if {'n_features_mean', 'macro_f1_mean'}.issubset(表.columns):
        图, 坐标=plt.subplots(figsize=(10,6))
        for _,行 in 表.iterrows():
            坐标.scatter(float(行['n_features_mean']), float(行['macro_f1_mean'])*100)
            坐标.annotate(方法中文(行[名称]), (float(行['n_features_mean']),float(行['macro_f1_mean'])*100),
                        xytext=(5,5), textcoords='offset points',fontsize=9)
        坐标.set(xlabel='五折平均特征数量（个）', ylabel='五折平均宏平均F1（%）',
               title='GRRF特征数量与分类性能的权衡')
        坐标.grid(alpha=.25)
        保存图(图,'05_特征数量与分类性能权衡.png')
    表 = 表.copy()
    表[名称] = 表[名称].map(方法中文)
    保存中文表(表, '03_gamma1公平参数搜索汇总.csv', {
        名称:'参数组合', 'n_features_mean':'平均特征数量',
        'n_features_std':'特征数量标准差', 'accuracy_mean':'准确率均值',
        'accuracy_std':'准确率标准差', 'macro_f1_mean':'宏平均F1均值',
        'macro_f1_std':'宏平均F1标准差',
        'selection_seconds_mean':'特征选择时间均值（秒）',
        'training_seconds_mean':'训练时间均值（秒）',
        'prediction_seconds_mean':'预测时间均值（秒）',
        'total_seconds_mean':'合计耗时均值（秒）'})


def 计算时间图():
    表 = 读取表('grrf_gamma1_grid_fair_svm_summary.csv')
    if 表 is None:return
    名称 = 寻找列(表, ['config_id'])
    列 = ['selection_seconds_mean','training_seconds_mean','prediction_seconds_mean']
    if not 名称 or not all(x in 表.columns for x in 列): return
    x=np.arange(len(表)); 底=np.zeros(len(表)); 图,坐标=plt.subplots(figsize=(11,6))
    for 字段, 标签 in zip(列,['RF与GRRF特征选择','SVM训练','SVM预测']):
        值 = 表[字段].astype(float).to_numpy()
        坐标.bar(x,值,bottom=底,label=标签)
        底 += 值
    坐标.set_xticks(x,[方法中文(v) for v in 表[名称]],rotation=17,ha='right')
    坐标.set(xlabel='参数组合',ylabel='每折平均耗时（秒）', title='特征选择、SVM训练与预测的时间成本')
    坐标.legend(); 坐标.grid(axis='y',alpha=.25)
    保存图(图,'06_计算耗时分析.png')


def SVM参数排错图():
    表=读取表('svm_preprocessing_diagnostic_summary.csv')
    if 表 is None:return
    名称=寻找列(表,['config'])
    if not 名称:return
    加图例条(表,名称,'accuracy_mean','macro_f1_mean',
       'SVM参数与标准化的影响（同一5折用户划分）','07_SVM参数诊断.png',12)
    表=表.copy()
    表[名称]=表[名称].map(方法中文)
    保存中文表(表,'04_SVM参数诊断汇总.csv',{
       名称:'SVM参数配置','accuracy_mean':'准确率均值','accuracy_std':'准确率标准差',
       'macro_f1_mean':'宏平均F1均值','macro_f1_std':'宏平均F1标准差'})


def 特征消融图():
    表=读取表('grrf_single_feature_ablation.csv')
    if 表 is not None and {'method','accuracy','macro_f1'}.issubset(表.columns):
        加图例条(表,'method','accuracy','macro_f1',
           '单特征消融：加入不同特征后的分类性能','08_单特征消融性能.png',12)
        表=表.copy();表['method']=表['method'].map(方法中文)
        保存中文表(表,'05_单特征消融结果.csv',{
            'method':'方法','n_features':'特征数量','accuracy':'准确率',
            'macro_f1':'宏平均F1','training_seconds':'训练时间（秒）',
            'prediction_seconds':'预测时间（秒）'})
    类别=读取表('grrf_single_feature_class_report.csv')
    if 类别 is None:return
    if not {'method','class_label','recall'}.issubset(类别.columns):return
    类别=类别[类别['method'].isin([
        'Common_6','Common_6_plus_gamma075_feature','Common_6_plus_gamma1_feature'])].copy()
    if 类别.empty:return
    表=类别.pivot(index='class_label',columns='method',values='recall')
    活动={'1':'走路','2':'上楼','3':'下楼','4':'坐','5':'站立','6':'躺下'}
    表.index=[活动.get(str(值),str(值)) for 值 in 表.index]
    表=表.rename(columns={列:方法中文(列) for 列 in 表.columns})
    图,坐标=plt.subplots(figsize=(12,6))
    表.plot(kind='bar',ax=坐标)
    坐标.set_ylim(0,1)
    坐标.set(xlabel='活动类别',ylabel='召回率',title='单特征消融：六类活动的召回率差异')
    坐标.tick_params(axis='x',rotation=0)
    坐标.legend(fontsize=8)
    坐标.grid(axis='y',alpha=.25)
    保存图(图,'09_单特征消融_活动召回率.png')
    保存中文表(表.reset_index().rename(columns={'index':'活动类别'}),
             '06_单特征消融_类别召回率.csv',{})


def 稳定性图():
    表=读取表('grrf_cv5_pairwise_jaccard.csv')
    if 表 is None:return
    相似度列=寻找列(表,['jaccard_similarity','jaccard','similarity'])
    γ列=寻找列(表,['gamma_grrf','gamma'])
    if not 相似度列:
        print('【提示】未识别Jaccard文件中的相似度列：',list(表.columns));return
    if γ列:
        汇总=表.groupby(γ列)[相似度列].agg(['mean','std','min','max']).reset_index()
        图,坐标=plt.subplots(figsize=(9,6))
        名称=['γ='+str(v) for v in 汇总[γ列]]
        坐标.bar(名称,汇总['mean'])
        坐标.set_ylim(0,1)
        坐标.set(xlabel='引导参数',ylabel='平均Jaccard相似度',
             title='不同训练用户划分下的特征选择稳定性')
        坐标.grid(axis='y',alpha=.25)
        保存图(图,'10_特征稳定性_Jaccard.png')
        保存中文表(汇总,'07_跨折特征稳定性.csv',{
          γ列:'GRRF引导参数γ','mean':'平均Jaccard相似度','std':'标准差',
          'min':'最小值','max':'最大值'})
    else:
        print('【提示】可读取Jaccard文件，但不能识别分组gamma列；跳过稳定性图。实际字段：',list(表.columns))


def 特征频次图():
    表=读取表('grrf_cv5_feature_frequency.csv')
    if 表 is None:return
    频次=寻找列(表,['n_folds','fold_frequency','frequency','count','selected_folds'])
    索引=寻找列(表,['feature_index_python','feature_index','feature_id'])
    参数=寻找列(表,['gamma_grrf','gamma'])
    if not 频次 or not 索引:
        print('【提示】特征频次表字段不能识别，实际字段：',list(表.columns));return
    # 分不同gamma绘图，避免把两种参数的选择频率混合成一个值。
    分组 = list(表.groupby(参数)) if 参数 else [('全部参数',表)]
    for 标记, 子表 in 分组:
        子表=子表.sort_values(频次,ascending=False).head(15)
        图,坐标=plt.subplots(figsize=(11,6))
        坐标.bar(子表[索引].astype(str),子表[频次].astype(float))
        坐标.set(xlabel='Python特征索引',ylabel='被选中的折数',
               title=f'GRRF特征选择频次（γ={标记}，前15个）')
        坐标.tick_params(axis='x',rotation=50)
        坐标.grid(axis='y',alpha=.25)
        保存图(图,f'11_特征选择频次_gamma_{标记}.png')


def 运行全部():
    print('开始生成中文版项目图表。不会修改原始 CSV，也不会重新训练模型。')
    研究流程图()
    参数敏感性图()
    五折核心对照()
    五折逐折图()
    论文参数网格图()
    计算时间图()
    SVM参数排错图()
    特征消融图()
    稳定性图()
    特征频次图()
    print(f'\n完成：生成{图数}张中文图片、{表数}份中文汇总表。')
    print('图片位置：', 图片目录)
    print('表格位置：', 中文表目录)
    print('请先在VS Code中打开图片检查，再提交到GitHub。')

if __name__=='__main__':
    运行全部()
