import org.apache.log4j.Logger
import org.apache.log4j.Level

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd._
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.util.MLUtils._
import org.apache.spark.mllib.regression.LabeledPoint

object TestSVM {

  def main(args: Array[String]) {

    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)	

    if (args.length != 1) {
      println("Usage: sbt/sbt package \"run libsvm-data\"")
      sys.exit(1)
    }

    // set up environment

    val jarFile = "target/scala-2.10/test-svm_2.10-0.0.jar"
    val conf = new SparkConf()
      .setMaster("local[2]")
      .setAppName("TestSVM")
      .set("spark.executor.memory", "2g")
      .setJars(Seq(jarFile))
    val sc = new SparkContext(conf)

    // load training data

    val dataDir = args(0)

    val data = loadLibSVMData(sc, dataDir).cache()

    val splits = data.randomSplit(Array(0.6, 0.4), seed = 0L)
    val training = splits(0).cache()
    val validation = splits(1).cache()

    val numTraining = training.count()
    val numValidation = validation.count()

    println("Training: " + numTraining + ", validation: " + numValidation)

    // Train models and evaluate them on the validation set

    val stepSizes = Seq(0.25, 0.5, 1.0, 2.0, 4.0)
    val numIters = Seq(10, 20, 40, 80, 160)
    val regParams = Seq(0.25, 0.5, 1.0, 2.0, 4.0)
    var bestModel: Option[SVMModel] = None
    var bestValidationAccuracy = Double.MinValue
    var bestStepSize = -1.0
    var bestRegParam = -1.0
    var bestNumIter = -1
    for (stepSize <- stepSizes; numIter <- numIters; regParam <- regParams) {
      val startTime = System.nanoTime()
      val model = SVMWithSGD.train(training, numIter, stepSize, regParam, 1.0)
      val time = (System.nanoTime() - startTime) / 1e9
      val validationAccuracy = computeAccuracy(model, validation, numValidation)
      println(s"""
          |Model trained with (numIter = $numIter, stepSize = $stepSize, regParam = $regParam) in $time seconds.
          |Model has accuracy of $validationAccuracy on validation.
        """.stripMargin)
      if (validationAccuracy > bestValidationAccuracy) {
        bestModel = Some(model)
        bestStepSize = stepSize
        bestRegParam = regParam
        bestNumIter = numIter
        bestValidationAccuracy = validationAccuracy
      }
    }

    println(s"""
          |The best model was trained with (numIter = $bestNumIter, stepSize = $bestStepSize, regParam = $bestRegParam).
          |The best model has accuracy of $bestValidationAccuracy on validation.
        """.stripMargin)

    sc.stop();
  }

  def computeAccuracy(model: SVMModel, data: RDD[LabeledPoint], n: Long) = {
    val predictions = model.predict(data.map(_.features))
    1.0 * data.map(_.label).zip(predictions).filter(x => x._1 == x._2).count() / n
  }
}
