package com.nimbus.demo;

import com.nimbus.fuzzybuckettree.FeatureConfig;
import com.nimbus.fuzzybuckettree.FuzzyBucketTree;
import com.nimbus.fuzzybuckettree.prediction.handlers.ClassificationPredictionHandler;
import com.nimbus.fuzzybuckettree.prediction.NodePrediction;

import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * Simple decisioning example with a String based classification result.
 * This is simple as can e and does not utilize any value bucketing or
 * auto-tuning features.
 */
public class SimpleExample {

    public static void main(String[] args) throws IOException {
        System.out.println("Hello world!");

        List<FeatureConfig> features = List.of(
            new FeatureConfig("size", 1, null),
            new FeatureConfig("color", 1, null),
            new FeatureConfig("legs", 1, null)
        );

        FuzzyBucketTree<String> tree = new FuzzyBucketTree<>(features, new ClassificationPredictionHandler<>());

        tree.train(Map.of("size", new float[]{"small".hashCode()}, "color",
                        new float[]{"brown".hashCode()}, "legs",
                        new float[]{4}),
                "dog");
        tree.train(Map.of("size", new float[]{"medium".hashCode()}, "color",
                        new float[]{"gray".hashCode()}, "legs",
                        new float[]{4}),
                "cat");
        tree.train(Map.of("size", new float[]{"large".hashCode()}, "color",
                        new float[]{"black".hashCode()}, "legs",
                        new float[]{4}),
                "horse");
        tree.train(Map.of("size", new float[]{"small".hashCode()}, "color",
                        new float[]{"white".hashCode()}, "legs",
                        new float[]{2}),
                "bird");
        tree.train(Map.of("size", new float[]{"large".hashCode()}, "color",
                        new float[]{"gray".hashCode()}, "legs",
                        new float[]{4}),
                "elephant");

        System.out.println("Trained");

        NodePrediction<String> p = tree.predict(Map.of("size", new float[]{"large".hashCode()}, "color",
                new float[]{"black".hashCode()}, "legs", new float[]{4}));

        if (p == null)
            System.out.println("No prediction");
        else
            System.out.println("Predicted label " + p.getPrediction() + " with confidence " + p.getConfidence() + " at depth " + p.getDepth());
    }

}