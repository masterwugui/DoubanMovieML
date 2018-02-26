package mrmodel;

import java.util.HashMap;
import java.util.Map;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.evaluation.RegressionMetrics;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.regression.LinearRegressionModel;
import org.apache.spark.mllib.regression.LinearRegressionWithSGD;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.RandomForestModel;

import scala.Tuple2;

public class LRModel {

	public static void main(String[] args) {
		SparkConf conf = new SparkConf().setAppName("JavaLRExample")
				.setMaster("local");
		//		.set("spark.testing.memory", "2147480000");
		JavaSparkContext sc = new JavaSparkContext(conf);
		sc.setLogLevel("ERROR");
		String trainingPath = "files/trainingData.txt";
		JavaRDD<LabeledPoint> trainingData = sc.textFile(trainingPath).map(line -> {
			String[] parts = line.split(" ");
			double[] v = new double[parts.length - 1];
			v[0] = Double.parseDouble(parts[1]);
			v[1] = Double.parseDouble(parts[2]);
			v[2] = Double.parseDouble(parts[3]);

			return new LabeledPoint(Double.parseDouble(parts[0]), Vectors.dense(v));
		});
		trainingData.cache();

		int numIterations = 5;
		LinearRegressionModel lrModel = LinearRegressionWithSGD.train(JavaRDD.toRDD(trainingData), numIterations);

		String testPath = "files/testData.txt";
		JavaRDD<LabeledPoint> testData = sc.textFile(testPath).map(line -> {
			String[] parts = line.split(" ");
			double[] v = new double[parts.length - 1];
			v[0] = Double.parseDouble(parts[1]);
			v[1] = Double.parseDouble(parts[2]);
			v[2] = Double.parseDouble(parts[3]);
			return new LabeledPoint(Double.parseDouble(parts[0]), Vectors.dense(v));
		});
		testData.cache();

		JavaPairRDD<Object, Object> predictionAndLabel = testData
				.mapToPair(p -> new Tuple2<>(lrModel.predict(p.features()), p.label()));

		RegressionMetrics metrics = new RegressionMetrics(predictionAndLabel.rdd());

		System.out.format("LR MSE = %f\n ", metrics.meanSquaredError());
		System.out.format("LR RMSE = %f\n ", metrics.rootMeanSquaredError());
		System.out.format("LR R Squared = %f\n ", metrics.r2());
		System.out.format("LR MAE = %f\n ", metrics.meanAbsoluteError());
		System.out.format("LR Explained Variance = %f\n ", metrics.explainedVariance());

		predictionAndLabel.collect()
				.forEach(p -> System.out.println("prediction: " + p._1 + " \t actual rating: " + p._2));
	}

}
