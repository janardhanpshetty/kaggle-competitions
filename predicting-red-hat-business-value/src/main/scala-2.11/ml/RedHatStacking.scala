package redhat.ml

import org.apache.spark.ml.classification.{GBTClassifier, RandomForestClassifier}
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, BinaryClassificationEvaluator}
import org.apache.spark.ml.feature.{HashingTF, VectorAssembler}
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Dataset, Row, SparkSession}

/**
 * Created by jshetty on 8/19/16.
 */
object RedHatStacking {
  // Udf's
  val trueFalseReplace = udf { (str: String) => {if (str == "true") 1 else 0}.toInt }
  val replaceNull = udf { (str: String) => if (str == "") "EMPTY" else str }
  val convertStringToArray = udf { (str: String) => Array(s"feature=$str") }
  val toDouble = udf[Double, String](_.toDouble)
  val toBin = udf { (str: String) => str.split(" ")(1).toInt match {
    case x if 0 until 10300 contains x => 1
    case x if 10300 until 20600 contains x => 2
    case x if 10300 until 20600 contains x => 3
    case x if 20600 until 30900 contains x => 4
    case x if 30900 until 52000 contains x => 5
  }
  }

  def getLabelFeatureDF(dataset: Dataset[Row], people: Dataset[Row]): Dataset[Row] = {
    // Rename train/test column names for joining
   val newCols = dataset.columns.map { x => (x + "_modified") }
    val colModifiedDF = dataset.toDF(newCols: _*)

    /* Pre-processing features of dataset
    * 1. Convert date columns of train/test to unixTimestamp
    * 2. Convert empty strings into Empty
     */
    val colModifiedUnDF = colModifiedDF.withColumn("date_modified", toDouble(unix_timestamp(colModifiedDF("date_modified"))))

    val cols = colModifiedUnDF.columns.toSeq
    val colModifiedReplaceDF = colModifiedUnDF.na.replace(cols, Map("" -> "EMPTY"))

    // Join two datasets using people_id
    val joinedDF = colModifiedReplaceDF.join(people, colModifiedReplaceDF("people_id_modified") === people("people_id")).drop(colModifiedReplaceDF("people_id_modified"))

    // Convert categorical variables into ArrayType as pre-req to HashingTF
    val nonHashCols = Seq("label_modified", "people_id", "activity_id_modified", "activity_category_modified", "date_modified", "date", "char_10", "char_38")
    val hashColumns = joinedDF.columns.diff(nonHashCols)

    var joinedArrayDF = joinedDF
    for (col <- hashColumns) {
      joinedArrayDF = joinedArrayDF.withColumn(col, convertStringToArray(joinedArrayDF(col)))
    }

    // Apply HashingTF to the columns
    val hashIndexers: Array[PipelineStage] = hashColumns.map { x => new HashingTF().setInputCol(x).setOutputCol(x + "hashTF").setNumFeatures(300) }
    val pipeline = new Pipeline().setStages(hashIndexers)
    val hashModelDF = pipeline.fit(joinedArrayDF).transform(joinedArrayDF)

    // Apply VectorAssembler to columns to get features
    val featuresCol = hashModelDF.columns.filter(x => x.matches(".*hashTF.*")) ++ Seq("char_38", "date_modified", "char_10")
    val vectorAssembler = new VectorAssembler().setInputCols(featuresCol).setOutputCol("features")
    val assemblerDF = vectorAssembler.transform(hashModelDF)
    assemblerDF
  }

  def main(args: Array[String]): Unit = {
    val trainFile = "/Users/jshetty/Documents/kaggle/redhat/dataset/act_train.csv"
    val peopleFile = "/Users/jshetty/Documents/kaggle/redhat/dataset/people.csv"
    val testFile = "/Users/jshetty/Documents/kaggle/redhat/dataset/act_test.csv"

    val spark = SparkSession.builder.master("local[8]").appName("RedHat Business-Value").getOrCreate()

    val trainInput = spark.read.option("header", "true").option("inferSchema", "true").csv(trainFile).cache()
    val peopleInput = spark.read.option("header", "true").option("inferSchema", "true").csv(peopleFile).cache()
    val testInput = spark.read.option("header", "true").option("inferSchema", "true").csv(testFile).cache()

    val train = trainInput.selectExpr("cast(outcome as double) as label", "date", "people_id", "activity_id")
    val peopleRaw = peopleInput.selectExpr("people_id", "date", "cast(char_38 as double) as char_38", "group_1", "char_7", "char_10")

    // Pre-processing People table
    val boolPeopleCols = Array("char_10")
    var peopleBool = peopleRaw
    for (col <- boolPeopleCols) {
      peopleBool = peopleBool.withColumn(col, trueFalseReplace(peopleBool(col)))
    }

    // convert date to unix
    val people = peopleBool.withColumn("date", unix_timestamp(peopleBool("date")))

    // Call getAssembler for train and people
    val labelFeatureDF = getLabelFeatureDF(train, people)
    val dataDF = labelFeatureDF.withColumnRenamed("label_modified", "label")

    //val assemblerDF = assembler.withColumnRenamed("label_train", "label")
    // Split dataset
    val Array(training, testing) = dataDF.randomSplit(Array(0.7, 0.3))

    // Fit the LR model
    //val lr = new LogisticRegression().setLabelCol("label").setMaxIter(100).setElasticNetParam(0.01)
    val rf = new RandomForestClassifier().setLabelCol("label").setFeaturesCol("features").setNumTrees(30)
    val gbt = new GBTClassifier().setLabelCol("label").setFeaturesCol("features").setPredictionCol("prediction").setMaxIter(30)

    //val rfModel = rf.fit(training)
    val gbtModel = gbt.fit(training)

    // Save the model
    //rfModel.save("/Users/jshetty/spark-applications/kaggle/models/redhat/Stackingrf")

    // Transform test for predictions
    val testPredictions = gbtModel.transform(testing)
    val trainPredictions = gbtModel.transform(training)

    // Evaluator
    val evaluator = new BinaryClassificationEvaluator().setMetricName("areaUnderROC")
    // Test ROC
    println("Area under ROC for test data = " + evaluator.evaluate(testPredictions))
    // Train ROC for checking over-fitting
    println("Area under ROC for training data = " + evaluator.evaluate(trainPredictions))


    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
    val accuracy = evaluator.evaluate(testPredictions)
    println("Test Error = " + (1.0 - accuracy))


    // For Submission
    val test = testInput.select("date", "people_id", "activity_id")
    val assemblerSubDF = getLabelFeatureDF(test, people)
    val testSubPredictions = gbtModel.transform(assemblerSubDF)
    println("Count of submission file = " + testSubPredictions.count)

    // Save submission file activity_id,outcome
    val outFile = testSubPredictions.select("activity_id_modified", "prediction").coalesce(1)
    outFile.write.csv("/Users/jshetty/Documents/kaggle/redhat/result/gbt")
    spark.stop()

  }
}