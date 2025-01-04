package com.nimbus.fuzzybuckettree.tuner.reporters;


import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;

public class ClassificationReporter<T> implements AccuracyReporter<T> {

    private final Map<T, PredictionReport> labelReports;
    private final Map<T, Float> classWeights;

    /**
     * Construct an accuracy reporter which records prediction results indexed by the correct
     * prediction values. Allows configurable class weights.
     * @param classWeights Map of prediction value to a weight multiplier e.g. <dog, 3.5f>
     */
    public ClassificationReporter(Map<T, Float> classWeights) {
        labelReports = new ConcurrentHashMap<>();
        this.classWeights = classWeights != null ? new ConcurrentHashMap<>(classWeights) : new ConcurrentHashMap<>();
    }

    /**
     * Construct a classification reporter where all labels have equal weight
     */
    public ClassificationReporter() {
        labelReports = new ConcurrentHashMap<>();
        this.classWeights = new ConcurrentHashMap<>();
    }

    @Override
    public boolean record(T predicted, T actual) {
        PredictionReport report = labelReports.computeIfAbsent(actual, voyd -> new PredictionReport());

        float weight = classWeights.computeIfAbsent(actual, voyd -> 1f);
        boolean match = predicted != null ? Objects.equals(actual, predicted) : false;
        report.increment(weight, match);

        return match;
    }

    @Override
    public float getTotalAccuracy() {
        return (float) labelReports.values().stream().mapToDouble(r -> r.calcWeightedAccuracy()).average().orElse(0f);
    }

    @Override
    public Map<T, PredictionReport> getAccuracyReports() {
        return labelReports;
    }

    @Override
    public ClassificationReporter<T> getNewInstance() {
        return new ClassificationReporter<>(classWeights);
    }

}
