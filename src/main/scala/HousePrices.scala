import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{Bucketizer, MinMaxScaler, OneHotEncoder, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.ml.regression.GBTRegressor
import org.apache.spark.ml.evaluation.RegressionEvaluator

object HousePrices {
  def main(args: Array[String]): Unit = {
    // rootPath为房价预测数据集根目录
    val rootPath: String = "./data"
    val filePath: String = s"$rootPath/train.csv"

    // 读取文件，创建DataFrame
    val spark = SparkSession
      .builder()
      .appName("sparkdf")
      .master("local[*]")
      .getOrCreate()

    var df: DataFrame = spark.read.format("csv").option("header", value = true).load(filePath)
    //df.printSchema

    //需要离散化的数值型字段
    var discreteFields = Array("BedroomAbvGr", "OverallQual", "OverallCond")

    //数值型字段
    val numericFields: Array[String] = Array("LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea" )

    //非数值型字段
    val categoricalFields: Array[String] = Array("MSSubClass", "MSZoning", "Street", "Alley", "LotShape", "LandContour", "Utilities", "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "YearBuilt", "YearRemodAdd", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "ExterQual", "ExterCond", "Foundation", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Heating", "HeatingQC", "CentralAir", "Electrical", "KitchenQual", "Functional", "FireplaceQu", "GarageType", "GarageYrBlt", "GarageFinish", "GarageQual", "GarageCond", "PavedDrive", "PoolQC", "Fence", "MiscFeature", "MiscVal", "MoSold", "YrSold", "SaleType", "SaleCondition")

    // Label字段
    val labelFields: Array[String] = Array("SalePrice")

    //把数值型字段从字符串转化为数字
    for (field <- (numericFields ++ labelFields ++ discreteFields)) {
      df = df
        .withColumn(s"${field}Int",col(field).cast(IntegerType))
        .drop(field)
    }

    // StringIndexer期望的输出列名
    val indexFields: Array[String] = categoricalFields.map(_ + "Index")
    // 定义StringIndexer实例
    val stringIndexer = new StringIndexer()
      .setInputCols(categoricalFields)
      .setOutputCols(indexFields)
      .setHandleInvalid("keep")

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
      .setSplitsArray(Array(OverallQualSplits, OverallCondSplits, bedroomAbvGrSplits))

    //热独编码
    //对所有非数值字段
    val oheFields: Array[String] = categoricalFields.map(_ + "OHE")
    val oneHotEncoder = new OneHotEncoder()
      .setInputCols(indexFields)
      .setOutputCols(oheFields)

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

    val gbtRegressor = new GBTRegressor()
      // 指定预测标的
      .setLabelCol("SalePriceInt")
      // 指定特征向量
      .setFeaturesCol("indexedFeatures")
      // 指定决策树的数量
      //.setMaxIter(200)
      // 指定决策树的最大深度
      //.setMaxDepth(10)
      .setMaxBins(113)

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
  }
}

