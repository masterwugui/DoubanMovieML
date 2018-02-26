package mrmodel;

import java.util.HashMap;
import java.util.Map;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.evaluation.RegressionMetrics;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.IsotonicRegression;
import org.apache.spark.mllib.regression.IsotonicRegressionModel;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.regression.LinearRegressionModel;
import org.apache.spark.mllib.regression.LinearRegressionWithSGD;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.rdd.RDD;

import scala.Tuple2;
import scala.Tuple3;

public class ISTOModel {

	public static void main(String[] args) {
		SparkConf conf = new SparkConf().setAppName("JavaISExample").setMaster("local");
		// .set("spark.testing.memory", "2147480000");
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

		JavaRDD<Tuple3<Double, Double, Double>> trainingData1 = trainingData
				.map(point -> new Tuple3<>(point.label(), point.features().apply(0), 1.0));
		IsotonicRegressionModel isModel = new IsotonicRegression().setIsotonic(true).run(trainingData1);

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
		JavaRDD<Tuple3<Double, Double, Double>> testData1 = testData
				.map(point -> new Tuple3<>(point.label(), point.features().apply(0), 1.0));
		
		JavaPairRDD<Object, Object> predictionAndLabel = testData1
				.mapToPair(p -> new Tuple2<>(isModel.predict(p._2()), p._1()));

		RegressionMetrics metrics = new RegressionMetrics(predictionAndLabel.rdd());

		System.out.format("IS MSE = %f\n ", metrics.meanSquaredError());
		System.out.format("IS RMSE = %f\n ", metrics.rootMeanSquaredError());
		System.out.format("IS R Squared = %f\n ", metrics.r2());
		System.out.format("IS MAE = %f\n ", metrics.meanAbsoluteError());
		System.out.format("IS Explained Variance = %f\n ", metrics.explainedVariance());

		predictionAndLabel.collect()
				.forEach(p -> System.out.println("prediction: " + p._1 + " \t actual rating: " + p._2));
	}

}
