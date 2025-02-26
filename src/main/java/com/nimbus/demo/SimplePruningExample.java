package com.nimbus.demo;

import com.nimbus.fuzzybuckettree.FeatureConfig;
import com.nimbus.fuzzybuckettree.FeatureValueType;
import com.nimbus.fuzzybuckettree.FuzzyBucketTree;
import com.nimbus.fuzzybuckettree.prediction.NodePrediction;
import com.nimbus.fuzzybuckettree.prediction.handlers.ClassificationPredictionHandler;
import com.nimbus.fuzzybuckettree.prediction.handlers.ExpiringClassificationHandler;

import java.io.IOException;
import java.time.Duration;
import java.util.List;
import java.util.Map;

/**
 * Simple decisioning example with a String based classification result.
 * This is simple as can e and does not utilize any value bucketing or
 * auto-tuning features.
 */
public class SimplePruningExample {

    public static void main(String[] args) throws IOException, InterruptedException {
        System.out.println("Hello world!");

        List<FeatureConfig> features = List.of(
            new FeatureConfig("size",  FeatureValueType.STRING,1, null),
            new FeatureConfig("color", FeatureValueType.STRING, 1, null),
            new FeatureConfig("legs",  FeatureValueType.INTEGER,1, null)
        );

        FuzzyBucketTree<String> tree = new FuzzyBucketTree<>(features,
                new ExpiringClassificationHandler<>(Duration.ofSeconds(2)), Duration.ofSeconds(1));

        tree.train(Map.of("size", new String[]{"small"}, "color",
                        new String[]{"brown"}, "legs",
                        new Integer[]{4}),
                "dog");
        tree.train(Map.of("size", new String[]{"medium"}, "color",
                        new String[]{"gray"}, "legs",
                        new Integer[]{4}),
                "cat");
        tree.train(Map.of("size", new String[]{"large"}, "color",
                        new String[]{"black"}, "legs",
                        new Integer[]{4}),
                "horse");
        tree.train(Map.of("size", new String[]{"small"}, "color",
                        new String[]{"white"}, "legs",
                        new Integer[]{2}),
                "bird");
        tree.train(Map.of("size", new String[]{"large"}, "color",
                        new String[]{"gray"}, "legs",
                        new Integer[]{4}),
                "elephant");

        System.out.println("Trained first");

        Thread.sleep(Duration.ofSeconds(1));

        tree.train(Map.of("size", new String[]{"small"}, "color",
                        new String[]{"white"}, "legs",
                        new Integer[]{4}),
                "mouse");
        tree.train(Map.of("size", new String[]{"large"}, "color",
                        new String[]{"gray"}, "legs",
                        new Integer[]{2}),
                "ostrich");

        NodePrediction<String> p = tree.predict(Map.of("size", new String[]{"large"}, "color",
                new String[]{"black"}, "legs", new Integer[]{4}));

        System.out.println(tree.leafCount());

        if (p == null)
            System.out.println("No prediction");
        else
            System.out.println("Predicted label " + p.getPrediction() + " with confidence " + p.getConfidence() + " at depth " + p.getDepth());

        Thread.sleep(Duration.ofSeconds(1));

        System.out.println(tree.leafCount());

        tree.saveModel("animals.fbt");
    }

}