# CCF2021 Default Prediction
代码思路 <br>
训练数据集中另一金融产品的违约数据，其中大部分特征与需要预测的数据特征相同，所以代码的提分点应该在于如何利用这些训练数据。<br>
伪标签 + 特征构造 + 树模型blending <br>
注意辅助数据中sub_class是更细粒度的特征，可以反向构造main data中sub_class这一强特征。
