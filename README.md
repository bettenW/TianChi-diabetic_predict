# TianChi-diabetic_predict
天池精准医疗大赛——人工智能辅助糖尿病遗传风险预测, 代码及总结
<br /> 
通过这次比赛进一步学习了数据分析以及sklearn应用的能力,也将平时的理论知识用于实践
<br /> 
## 整体思路
> 1.在预测血糖是发现模型对于高血糖预测不敏感,所以事先通过分类预测数据是高血糖和低血糖的概率.然后对模型进行加权
>
>2.具体加权,首先将训练集分为高血糖和低血糖两类,分别拟合模型.然后对相同的测试集预测,分别乘以高血糖和低血糖的概率即可
## 参考代码
> 1. Regressor模型融合 https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
>
> 2. Classification模型融合 https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling
