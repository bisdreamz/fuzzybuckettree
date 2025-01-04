package com.nimbus.fuzzybuckettree.tuner.reporters;

import java.util.Map;

public class RegressionAccuracyReporter implements AccuracyReporter<Float> {

    private float correctSum;
    private float predictedSum;

    @Override
    public boolean record(Float predicted, Float actual) {
        correctSum += actual;
        predictedSum += predicted;

        return Float.compare(predictedSum, correctSum) == 0;
    }

    @Override
    public float getTotalAccuracy() {
        if (correctSum == 0f || predictedSum == 0f)
            return 0f;

        return Math.min(predictedSum, correctSum) / Math.max(predictedSum, correctSum);
    }

    @Override
    public Map getAccuracyReports() {
        return Map.of("all", getTotalAccuracy());
    }

    @Override
    public AccuracyReporter getNewInstance() {
        return new RegressionAccuracyReporter();
    }

}
