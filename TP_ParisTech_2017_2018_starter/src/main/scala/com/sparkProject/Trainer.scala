package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{CountVectorizer, RegexTokenizer, StopWordsRemover}
import org.apache.spark.ml.feature.{IDF, StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression}

object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()

    /*******************************************************************************
      *
      *       TP 4-5
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

   /** CHARGER LE DATASET **/


   val data = spark.read.parquet("prepared_trainingset")

    /** TF-IDF **/

        /** 1er stage: Tokenizer **/

    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    //val words=tokenizer.transform(data)

        /** 2eme stage: Retirer les stop words **/

    val remover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("filtered")

    //val words_clean = remover.transform(words)

        /** 3eme stage: TF-IDF **/

    val cv = new CountVectorizer()
      .setInputCol(remover.getOutputCol)
      .setOutputCol("countVectorized")


    //val cvModel = cv.fit(words_clean)

    //val d = cvModel.transform(words_clean)

        /** 4eme stage: Trouver la partie IDF **/

    val idf = new IDF().setInputCol("countVectorized").setOutputCol("tfidf")

    //val idfModel = idf.fit(d)

    //val tfidf = idfModel.transform(d)

        /** 5eme stage : Indexer les pays **/

    val indexer = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")

    //val indexed = indexer.fit(tfidf).transform(tfidf)

        /** 6eme stage: Indexer les currencies **/

    val indexer2 = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")

    //val indexed2 = indexer2.fit(indexed).transform(indexed)

    /** VECTOR ASSEMBLER **/

    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf","days_campaign", "hours_prepa","goal","country_indexed","currency_indexed"))
      .setOutputCol("features")

    //val output = assembler.transform(indexed2)

    //output.select("features").show()

    /** MODEL **/

    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(300)


    /** PIPELINE **/

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, cv, idf, indexer, indexer2, assembler, lr))

    /** TRAINING AND GRID-SEARCH **/

    val Array(trainingData, testData) = data.randomSplit(Array(0.9, 0.1))

    val paramGrid = new ParamGridBuilder()
      .addGrid(cv.minDF,Array(55.0,75.0,95.0))
      .addGrid(lr.regParam, Array(0.00000001, 0.000001,0.0001,0.01))
      .build()

    val mce = new MulticlassClassificationEvaluator()
      .setPredictionCol("predictions")
      .setLabelCol("final_status")
      .setMetricName("f1")


    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(mce)
      .setEstimatorParamMaps(paramGrid)
      // 70% of the data will be used for training and the remaining 30% for validation.
      .setTrainRatio(0.7)


    val model = trainValidationSplit.fit(trainingData)

    // Make predictions on test data. model is the model with combination of parameters
    // that performed best.

    val df_WithPredictions = model.transform(testData)

    val f1=mce.evaluate(df_WithPredictions)
    print("f1 score is " + f1)

    df_WithPredictions.select("final_status","predictions").show()

    df_WithPredictions.groupBy("final_status", "predictions").count.show()

    model.write.overwrite().save("myModel")

  }
}
