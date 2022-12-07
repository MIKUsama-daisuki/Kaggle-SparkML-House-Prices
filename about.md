# 写在前面

数据集：[House Prices - Advanced Regression Techniques | Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview)

参考：[零基础入门 Spark (geekbang.org)](https://time.geekbang.org/column/intro/100090001?utm_term=iTab&utm_source=iTab&utm_medium=iTab&utm_campaign=iTab&utm_content=iTab&tab=catalog)

# 具体实现

## 特征工程

### 1. 读取数据

```scala
val rootPath: String = _
val filePath: String = s"$rootPath/train.csv"
// 读取文件，创建DataFrame
val spark = SparkSession
  .builder()
  .appName("sparkdf")
  .master("local[*]")
  .getOrCreate()
var df: DataFrame = spark.read.format("csv").option("header", value = true).load(filePath)
```

### 2. 数据分类

​		为了提升数据的区分度，对部分字段采用离散化处理，所以要事先分配好需要离散化的字段

```scala
//需要离散化的字段
var discreteFields =Array("BedroomAbvGr", "OverallQual", "OverallCond")
//数值型字段
val numericFields: Array[String] = Array("LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea" )
//非数值型字段
val categoricalFields: Array[String] = Array("MSSubClass", "MSZoning", "Street", "Alley", "LotShape", "LandContour", "Utilities", "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "YearBuilt", "YearRemodAdd", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "ExterQual", "ExterCond", "Foundation", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Heating", "HeatingQC", "CentralAir", "Electrical", "KitchenQual", "Functional", "FireplaceQu", "GarageType", "GarageYrBlt", "GarageFinish", "GarageQual", "GarageCond", "PavedDrive", "PoolQC", "Fence", "MiscFeature", "MiscVal", "MoSold", "YrSold", "SaleType", "SaleCondition")
// Label字段
val labelFields: Array[String] = Array("SalePrice")
```

### 3. 预处理：StringIndexer

> 官网对StringIndexer的描述：
>
> A label indexer that maps string column(s) of labels to ML column(s) of label indices. If the input columns are numeric, we cast them to string and index the string values. The indices are in [0, numLabels). By default, this is ordered by label frequencies so the most frequent label gets index 0. The ordering behavior is controlled by setting stringOrderType.

简单来说就是以数据列为单位，把字段中的字符串转换为数值索引。

```scala
//把数值型字段从字符串转化为数字
for (field <- (numericFields ++ labelFields ++ discreteFields)) {
  df = df
    .withColumn(s"${field}Int",col(field).cast(IntegerType))
    .drop(field)
}
// 用StringIndexer把非数值字段转化为数值字段
val indexFields: Array[String] = categoricalFields.map(_ + "Index")
// 定义StringIndexer实例
val stringIndexer = new StringIndexer()
  .setInputCols(categoricalFields)
  .setOutputCols(indexFields)
  .setHandleInvalid("keep")
```

### 4. 离散化：Bucketizer

离散化的目的：

​	对于BedroomAbvGr这个字段，它的含义是居室数量，在 train.csv 这份数据样本中，“BedroomAbvGr”包含从 1 到 8 的连续整数。我们可以将1，2分为小户型，3，4分为中户型，5，6，7，8分为大户型，这样离散化后，数据的多样性从原来的8降低为3。离散化的动机，主要在于提升特征数据的区分度与内聚性，从而与预测标的产生更强的关联。

> 官网对Bucketizer的描述：
>
> Bucketizer maps a column of continuous features to a column of feature buckets.
>
> Since 2.3.0, Bucketizer can map multiple columns at once by setting the inputCols parameter. Note that when both the inputCol and inputCols parameters are set, an Exception will be thrown. The splits parameter is only used for single column usage, and splitsArray is for multiple columns.

简单来说就是将连续特征映射为特征桶（feature buckets）

我选取了三个特征做离散化操作，以下是这三个特征的描述：

```scala
OverallQual: Rates the overall material and finish of the house
       10	Very Excellent
       9	Excellent
       8	Very Good
       7	Good
       6	Above Average
       5	Average
       4	Below Average
       3	Fair
       2	Poor
       1	Very Poor	
OverallCond: Rates the overall condition of the house
       10	Very Excellent
       9	Excellent
       8	Very Good
       7	Good
       6	Above Average	
       5	Average
       4	Below Average	
       3	Fair
       2	Poor
       1	Very Poor
BedroomAbvGr: Number of bedrooms
```

对于前两个特征OverallQual和OverallCond，我将它们从3和7的位置分隔为3个等级

第三个特征BedroomAbvGr在数据集中的值域是[1,8]，我将它从3和5的位置分为3个等级

```scala
//离散化
val discrete = discreteFields.map(_ + "Discrete")
discreteFields = discreteFields.map(_ + "Int")
val bedroomAbvGrSplits: Array[Double] = Array(Double.NegativeInfinity, 3, 5, Double.PositiveInfinity)
val OverallQualSplits: Array[Double] = Array(Double.NegativeInfinity, 3, 7, Double.PositiveInfinity)
val OverallCondSplits: Array[Double] = Array(Double.NegativeInfinity, 3, 7, Double.PositiveInfinity)
// 定义并初始化Bucketizer
val bucketizer = new Bucketizer()
  // 指定原始列
  .setInputCols(discreteFields)
  // 指定目标列
  .setOutputCols(discrete)
  // 指定离散区间
  .setSplitsArray(Array(OverallQualSplits,OverallCondSplits, bedroomAbvGrSplits))
```

### 5. Embedding：OneHotEncoder

对于字段值不存在大小关系的字段来说，只是将其转化为数值型字段是没有意义的，这就要用到Embedding（向量化）了

> 官网对OneHotEncoder的描述：
>
> A one-hot encoder that maps a column of category indices to a column of binary vectors, with at most a single one-value per row that indicates the input category index. For example with 5 categories, an input value of 2.0 would map to an output vector of [0.0, 0.0, 1.0, 0.0]. The last category is not included by default (configurable via dropLast), because it makes the vector entries sum up to one, and hence linearly dependent. So an input value of 4.0 maps to [0.0, 0.0, 0.0, 0.0].

```scala
//热独编码
//对所有非数值字段
val oheFields: Array[String] = categoricalFields.map(_ + "OHE")
val oneHotEncoder = new OneHotEncoder()
  .setInputCols(indexFields)
  .setOutputCols(oheFields)
```

### 6. 归一化：MinMaxScaler

归一化（Normalization）的作用，是把一组数值，统一映射到同一个值域，而这个值域通常是[0, 1]。当原始数据之间的量纲差异较大时，在模型训练的过程中，梯度下降不稳定、抖动较大，模型不容易收敛，从而导致训练效率较差。相反，当所有特征数据都被约束到同一个值域时，模型训练的效率会得到大幅提升。

> 官网对OneHotEncoder的描述：
>
> Rescale each feature individually to a common range [min, max] linearly using column summary statistics, which is also known as min-max normalization or Rescaling. The rescaled value for feature E is calculated as:
>
> ![](https://static001.geekbang.org/resource/image/2c/2e/2c934e54729931bdf18288c93040a42e.png?wh=744x104)

```scala
// 构建特征向量
val numericFeatures: Array[String] = numericFields.map(_ + "Int")
val vectorAssembler = new VectorAssembler()
  .setInputCols(numericFeatures ++ indexFields ++ discrete)
  .setOutputCol("features")
  .setHandleInvalid("keep")
//归一化
val minMaxScaler = new MinMaxScaler()
  .setInputCol("features")
  .setOutputCol("vector")
val vectorIndexer = new VectorIndexer()
  .setInputCol("vector")
  .setOutputCol("indexedFeatures")
  // 指定连续、离散判定阈值
  .setMaxCategories(30)
  .setHandleInvalid("keep")
```

## 决策树模型

用不同特征的不同取值的组合，将数据集分为若干类

> 维基百科对决策树的描述：
>
> Gradient boosting is a machine learning technique used in regression and classification tasks, among others. It gives a prediction model in the form of an ensemble of weak prediction models, which are typically decision trees. When a decision tree is the weak learner, the resulting algorithm is called gradient-boosted trees; it usually outperforms random forest.A gradient-boosted trees model is built in a stage-wise fashion as in other boosting methods, but it generalizes the other methods by allowing optimization of an arbitrary differentiable loss function.

```scala
val gbtRegressor = new GBTRegressor()
  // 指定预测标的
  .setLabelCol("SalePriceInt")
  // 指定特征向量
  .setFeaturesCol("indexedFeatures")
  // 指定决策树的数量
  .setMaxIter(30)
  // 指定决策树的最大深度
  .setMaxDepth(5)
  .setMaxBins(113)
```

## 模型训练与测试

```scala
//将所有步骤拼接起来
val components = Array(stringIndexer, bucketizer, oneHotEncoder, vectorAssembler, minMaxScaler, vectorIndexer, gbtRegressor)
val pipeline = new Pipeline().setStages(components)
// 划分出训练集和验证集
val Array(trainingData, validationData) = df.randomSplit(Array(0.7, 0.3))
// 调用fit方法，触发Pipeline计算，并最终拟合出模型
val pipelineModel = pipeline.fit(trainingData)

//测试
val predictions: DataFrame = pipelineModel.transform(validationData).select("SalePriceInt", "prediction")
predictions.show
val evaluator = new RegressionEvaluator().setLabelCol("SalePriceInt").setPredictionCol("prediction").setMetricName("rmse")
val rmse = evaluator.evaluate(predictions)
println("Root Mean Squared Error (RMSE) on test data = " + rmse)

//输出
+------------+------------------+
|SalePriceInt|        prediction|
+------------+------------------+
|      208500| 194936.7390633201|
|      118000| 119025.8916425119|
|       82000| 61752.41121907966|
|       86000| 66919.40597545142|
|      232000|247479.64950853962|
|      205000|185552.29830149628|
|      102000|127362.53595460855|
|      227000|208367.37768488855|
|      203000|  238802.515083912|
|      178000| 166284.1585451415|
|      191000|190125.10112212482|
|      287000| 423247.5498783909|
|      112500|128910.00291716766|
|      293077|240677.67824324753|
|       84000|  61818.4933553084|
|      315500| 263464.3610693604|
|       80000| 78665.99046630661|
|      262280| 302944.9443747723|
|      139600|175819.03135313356|
|      169500|205292.33446646377|
+------------+------------------+
only showing top 20 rows
Root Mean Squared Error (RMSE) on test data = 36637.90806077914
```

