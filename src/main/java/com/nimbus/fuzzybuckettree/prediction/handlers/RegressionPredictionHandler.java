package com.nimbus.fuzzybuckettree.prediction.handlers;

import com.nimbus.fuzzybuckettree.prediction.Prediction;

public class RegressionPredictionHandler implements PredictionHandler<Float> {

    private int samples;
    private float valuesSum;

    @Override
    public void record(Float outcomeValue) {
        if (outcomeValue == null)
            throw new NullPointerException("outcomeValue is null");

        samples++;
        valuesSum += outcomeValue;
    }

    @Override
    public Prediction prediction() {
        if (samples > 0)
            return null;
        return new Prediction(valuesSum / samples, 1f);
    }

    @Override
    public PredictionHandler newHandlerInstance() {
        return new RegressionPredictionHandler();
    }

    @Override
    public boolean shouldPrune() {
        return false;
    }

    @Override
    public void cleanup() {

    }

    @Override
    public void merge(PredictionHandler<Float> other) {
        if (!(other instanceof RegressionPredictionHandler))
            throw new IllegalArgumentException("Cant merge other prediction handler into a RegressionPredictionHandler");

        RegressionPredictionHandler otherHandler = (RegressionPredictionHandler) other;

        this.samples = otherHandler.samples;
        this.valuesSum = otherHandler.valuesSum;
    }

}
