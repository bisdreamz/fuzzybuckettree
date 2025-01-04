package com.nimbus.demo;

import com.nimbus.fuzzybuckettree.FuzzyBucketTree;
import com.nimbus.fuzzybuckettree.prediction.NodePrediction;
import com.nimbus.fuzzybuckettree.tuner.TrainingEntry;
import com.nimbus.fuzzybuckettree.tuner.reporters.AccuracyReporter;
import com.nimbus.fuzzybuckettree.tuner.reporters.ClassificationReporter;

import java.io.IOException;
import java.util.List;

public class CarCsvFromFile {

    public static void main(String[] args) throws IOException {
        FuzzyBucketTree<String> model = FuzzyBucketTree.loadModel("cars.fbt");
        List<TrainingEntry<String>> data = DemoUtils.loadCarData();

        AccuracyReporter<String> reporter = new ClassificationReporter<>();
        data.forEach(en -> {
            NodePrediction<String> p = model.predict(en.features());
            reporter.record(p.getPrediction(), en.outcome());
        });

        System.out.println("Achieved final accuracy on file loaded model of " + reporter.getTotalAccuracy());
        reporter.getAccuracyReports().forEach((k, r) -> {
            System.out.println(k + " -> " + r.getCorrect() + " ? " + r.getSamples());
        });
    }
}
