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
import org.apache.spark.mllib.tree.GradientBoostedTrees;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.configuration.BoostingStrategy;
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel;
import org.apache.spark.mllib.tree.model.RandomForestModel;

import scala.Tuple2;

public class GBDTModel {

	public static void main(String[] args) {
		SparkConf conf = new SparkConf().setAppName("JavaGBDTxample").setMaster("local");
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

		BoostingStrategy boostingStrategy = BoostingStrategy.defaultParams("Regression");
		boostingStrategy.setNumIterations(10);
		boostingStrategy.getTreeStrategy().setMaxDepth(5);
		Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
		boostingStrategy.treeStrategy().setCategoricalFeaturesInfo(categoricalFeaturesInfo);
		GradientBoostedTreesModel GBDTModel = GradientBoostedTrees.train(trainingData, boostingStrategy);
		
		
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
				.mapToPair(p -> new Tuple2<>(GBDTModel.predict(p.features()), p.label()));

		RegressionMetrics metrics = new RegressionMetrics(predictionAndLabel.rdd());

		System.out.format("GBDT MSE = %f\n ", metrics.meanSquaredError());
		System.out.format("GBDT RMSE = %f\n ", metrics.rootMeanSquaredError());
		System.out.format("GBDT R Squared = %f\n ", metrics.r2());
		System.out.format("GBDT MAE = %f\n ", metrics.meanAbsoluteError());
		System.out.format("GBDT Explained Variance = %f\n ", metrics.explainedVariance());

		predictionAndLabel.collect()
				.forEach(p -> System.out.println("prediction: " + p._1 + " \t actual rating: " + p._2));
	}

}
