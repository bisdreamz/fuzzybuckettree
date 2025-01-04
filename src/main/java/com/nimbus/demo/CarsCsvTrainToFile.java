package com.nimbus.demo;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.nimbus.fuzzybuckettree.FuzzyBucketTuner;
import com.nimbus.fuzzybuckettree.prediction.NodePrediction;
import com.nimbus.fuzzybuckettree.prediction.handlers.ClassificationPredictionHandler;
import com.nimbus.fuzzybuckettree.tuner.FeatureBucketOptions;
import com.nimbus.fuzzybuckettree.tuner.TrainingEntry;
import com.nimbus.fuzzybuckettree.tuner.TunerResult;
import com.nimbus.fuzzybuckettree.tuner.reporters.AccuracyReporter;
import com.nimbus.fuzzybuckettree.tuner.reporters.ClassificationReporter;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;

public class CarsCsvTrainToFile {

    public static void main(String[] args) throws IOException {
        List<TrainingEntry<String>> csvData = DemoUtils.loadCarData();

        List<FeatureBucketOptions> featureBucketOptions = List.of(
                new FeatureBucketOptions("price", 1,null),
                new FeatureBucketOptions("maint", 1,null),
                new FeatureBucketOptions("doors",1, null),
                new FeatureBucketOptions("persons", 1,null),
                new FeatureBucketOptions("lugboot",1, null),
                new FeatureBucketOptions("safety",1, null)
        );

        FuzzyBucketTuner<String> tuner = new FuzzyBucketTuner<>(featureBucketOptions,
                new ClassificationPredictionHandler<>(),
                new ClassificationReporter<>());

        List<TrainingEntry<String>> validationData = new ArrayList<>();
        for (TrainingEntry<String> trainingEntry : csvData) {
            if (ThreadLocalRandom.current().nextInt(100) < 20)
                validationData.add(trainingEntry);
        }

        TunerResult<String> result = tuner.train(csvData, validationData);
        System.out.println("Achieved final accuracy of " + result.getTotalAccuracy());
        result.getAccuracyReports().forEach((k, r) -> {
            System.out.println(k + " -> " + r.getCorrect() + " ? " + r.getSamples());
        });

        System.out.println("Feature Configs");
        result.getFeatureConfigs().forEach(fc -> {
            System.out.println(fc.label() + " -> " + Arrays.toString(fc.buckets()));
        });

        try {
            //System.out.println("Training data: " + new ObjectMapper().writeValueAsString(csvData));
            //System.out.println("Tree state: " + new ObjectMapper().writeValueAsString(result.getTree()));
        } catch (Throwable e) {
            e.printStackTrace();
        }

        AccuracyReporter<String> reporter = new ClassificationReporter<>();
        validationData.forEach(vde -> {
            NodePrediction<String> p = result.getTree().predict(vde.features());
            boolean correct = reporter.record(p.getPrediction(), vde.outcome());

            if (!correct) {
                System.out.println("Incorrect - predicted " + p.getPrediction() + " actual " + vde.outcome()
                        + ": " + vde.features().entrySet().stream().map(en -> en.getKey() + "=" + Arrays.toString(en.getValue()))
                        .collect(Collectors.joining(",")));
            }
        });

        result.getTree().saveModel("cars.fbt");

        System.out.println("Done!");
    }

}
