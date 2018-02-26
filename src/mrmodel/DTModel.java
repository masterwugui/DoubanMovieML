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
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.mllib.tree.model.RandomForestModel;

import scala.Tuple2;

public class DTModel {

	public static void main(String[] args) {
		SparkConf conf = new SparkConf().setAppName("JavaDTExample").setMaster("local");
		JavaSparkContext sc = new JavaSparkContext(conf);
		sc.setLogLevel("ERROR");
		String trainingPath = "files/trainingData.txt";
		JavaRDD<LabeledPoint> trainingData = sc.textFile(trainingPath).map(line -> {
			String[] parts = line.split(" ");
			double[] v = new double[parts.length - 1];
			// for (int i = 1; i < parts.length; i++) {
			// v[i - 1] = Double.parseDouble(parts[i]);
			// }
			v[0] = Double.parseDouble(parts[1]);
			v[1] = Double.parseDouble(parts[2]);
			v[2] = Double.parseDouble(parts[3]);

			return new LabeledPoint(Double.parseDouble(parts[0]), Vectors.dense(v));
		});
		trainingData.cache();

		Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
		String impurity = "variance";
		int maxDepth = 5;
		int maxBins = 32;
		DecisionTreeModel dtModel = DecisionTree.trainRegressor(trainingData, categoricalFeaturesInfo, impurity,
				maxDepth, maxBins);

		String testPath = "files/testData.txt";
		JavaRDD<LabeledPoint> testData = sc.textFile(testPath).map(line -> {
			String[] parts = line.split(" ");
			double[] v = new double[parts.length - 1];
			// for (int i = 1; i < parts.length; i++) {
			// v[i - 1] = Double.parseDouble(parts[i]);
			// }
			v[0] = Double.parseDouble(parts[1]);
			v[1] = Double.parseDouble(parts[2]);
			v[2] = Double.parseDouble(parts[3]);
			return new LabeledPoint(Double.parseDouble(parts[0]), Vectors.dense(v));
		});
		testData.cache();

		JavaPairRDD<Object, Object> predictionAndLabel = testData
				.mapToPair(p -> new Tuple2<>(dtModel.predict(p.features()), p.label()));

		RegressionMetrics metrics = new RegressionMetrics(predictionAndLabel.rdd());

		System.out.format("DT MSE = %f\n ", metrics.meanSquaredError());
		System.out.format("DT RMSE = %f\n ", metrics.rootMeanSquaredError());
		System.out.format("DT R Squared = %f\n ", metrics.r2());
		System.out.format("DT MAE = %f\n ", metrics.meanAbsoluteError());
		System.out.format("DT Explained Variance = %f\n ", metrics.explainedVariance());

		predictionAndLabel.collect()
				.forEach(p -> System.out.println("prediction: " + p._1 + " \t actual rating: " + p._2));
	}

}
