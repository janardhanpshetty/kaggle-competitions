package redhat.ml

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, VectorAssembler}
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Dataset, Row, SparkSession}

/**
 * Created by jshetty on 8/19/16.
 */
object RedHatBaseline {

  // Udf's
  val trueFalseReplace = udf { (str: String) => if (str == "true") 1 else 0}
  val replaceNull = udf { (str: String) => if (str == "") "EMPTY" else str }
  val convertStringToArray = udf{(str: String) => Array(s"feature=$str")}

  def getAssembler(dataset: Dataset[Row], people: Dataset[Row]): Dataset[Row] = {
    // Rename column names of train/test to help in join
    val newCols = dataset.columns.map { x => (x + "_train") }
    val colModifiedDataset = dataset.toDF(newCols: _*)

    // Join two datasets using people_id
    val joinedDF = colModifiedDataset.join(people, colModifiedDataset("people_id_train") === people("people_id")).drop(colModifiedDataset("people_id_train"))

    // Convert Date columns into unixTimestamp
    val dateColumns = Array("date_train", "date")
    var joinedDFUnix = joinedDF
    for (col <- dateColumns) {
      joinedDFUnix = joinedDFUnix.withColumn(col, unix_timestamp(joinedDFUnix(col)))
    }
    joinedDFUnix.show(4)

    // Convert null columns to a string named EMPTY
    val nullColumns = joinedDFUnix.columns.filter(x => x.matches(".*char.*"))
    var joinedDFUnixNull = joinedDFUnix
    for (col <- nullColumns) {
      joinedDFUnixNull = joinedDFUnixNull.withColumn(col, replaceNull(joinedDFUnixNull(col)))
    }
    joinedDFUnixNull.show(4)

    // Change to ArrayType
    val nonHashingCols = Array("activity_id_train", "date_train", "label_train", "char_38", "date", "people_id")
    val hasherColumns = joinedDFUnixNull.columns.diff(nonHashingCols)
    var joinedDFUnixNullArray = joinedDFUnixNull
    for (col <- hasherColumns) {
      joinedDFUnixNullArray = joinedDFUnixNullArray.withColumn(col, convertStringToArray(joinedDFUnixNullArray(col)))
    }
    joinedDFUnixNullArray.show(4)

    // --- End of preprocessing

    // Apply HashingTF to the columns
    val hasherIndexers: Array[PipelineStage] = hasherColumns.map { x => new HashingTF().setInputCol(x).setOutputCol(x + "_hasherTF").setNumFeatures(2048) }
    val pipeline1 = new Pipeline().setStages(hasherIndexers)
    val hasherModelDF = pipeline1.fit(joinedDFUnixNullArray).transform(joinedDFUnixNullArray)

    // Apply VectorAssembler to columns to get features
    val featuresCol = hasherModelDF.columns.filter(x => x.matches(".*hasherTF.*"))
    val assemblerf = new VectorAssembler().setInputCols(featuresCol).setOutputCol("features")
    val assembler = assemblerf.transform(hasherModelDF)
    assembler
  }

  def main(args: Array[String]): Unit = {

    if (args.length < 4) {
      System.err.println("Usage: <train_file> <people_file> <test_file> <output_file>")
      System.exit(1)
    }
    val trainFile = args(0)
    val peopleFile = args(1)
    val testFile = args(2)
    val outputFile = args(3)

    val spark = SparkSession.builder.master("local[8]").appName("RedHat Business-Value").getOrCreate()
    val trainf = spark.read.option("header", "true").option("inferSchema", "true").csv(trainFile).cache()
    val peoplef = spark.read.option("header", "true").option("inferSchema", "true").csv(peopleFile).cache()
    val testf = spark.read.option("header", "true").option("inferSchema", "true").csv(testFile).cache()

    val train = trainf.selectExpr("cast(outcome as double) as label", "people_id", "activity_id", "date", "activity_category", "char_1", "char_2", "char_3", "char_4", "char_5", "char_6", "char_7", "char_8", "char_9", "char_10")

    // Replace true false columns with binary 0 and 1
    val booleanPeopleColumns = Array("char_10","char_11","char_12","char_13","char_14","char_15","char_16","char_17","char_18","char_19","char_20","char_21","char_22","char_23","char_24","char_25","char_26","char_27","char_28","char_29","char_30","char_31","char_32","char_33","char_34","char_35","char_36","char_37")
    var processedPeople = peoplef
    for(col <- booleanPeopleColumns){
      processedPeople = processedPeople.withColumn(col,trueFalseReplace(processedPeople(col)))
    }

    // Call getAssembler for train and people
    val getAssemblerDF = getAssembler(train, processedPeople)
    val assemblerDF = getAssemblerDF.withColumnRenamed("label_train", "label")

    // Split dataset
    val Array(training, testing) = assemblerDF.randomSplit(Array(0.8, 0.2))

    // Apply machine learning
    val lr = new LogisticRegression().setLabelCol("label")
    val pipelineLR = new Pipeline().setStages(Array(lr))
    // Model
    val lrModel = pipelineLR.fit(training)

    // Save the model
    lrModel.save("/Users/jshetty/spark-applications/kaggle/models/redhat/lrModelHashing")

    // Transform test for predictions
    val testPredictions = lrModel.transform(testing)
    val trainPredictions = lrModel.transform(training)

    // Evaluator
    val evaluator = new BinaryClassificationEvaluator().setMetricName("areaUnderROC")
    // Test ROC
    println("Area under ROC for test data = " + evaluator.evaluate(testPredictions))
    // Train ROC for checking overfitting
    println("Area under ROC for training data = " + evaluator.evaluate(trainPredictions))

    testPredictions.show(4)

    // For Submission
    val assemblerSubDF = getAssembler(testf, processedPeople)
    val testSubPredictions = lrModel.transform(assemblerSubDF)
    println("Count of submission file = " + testSubPredictions.count)

    // Save submission file activity_id,outcome
    val outFile = testSubPredictions.select("activity_id_train", "prediction").coalesce(1)
    outFile.write.csv(args(3))

    spark.stop()

  }
}