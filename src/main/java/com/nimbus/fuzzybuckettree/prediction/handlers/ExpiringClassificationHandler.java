package com.nimbus.fuzzybuckettree.prediction.handlers;

import com.nimbus.fuzzybuckettree.prediction.Prediction;
import com.nimbus.fuzzybuckettree.tuner.reporters.ClassificationReporter;

import java.time.Duration;
import java.time.Instant;

public class ExpiringClassificationHandler<T> implements PredictionHandler<T> {

    private final ClassificationPredictionHandler<T> handler;
    private final Duration expiryDuration;
    private Instant expiry;

    public ExpiringClassificationHandler(Duration expireAfter) {
        this.handler = new ClassificationPredictionHandler<>();
        this.expiryDuration = expireAfter;
        this.expiry = Instant.now().plus(expireAfter);
    }

    @Override
    public void record(T outcomeValue) {
        this.expiry = Instant.now().plus(this.expiryDuration);
        this.handler.record(outcomeValue);
    }

    @Override
    public void merge(PredictionHandler other) {
        handler.merge(other);
        ExpiringClassificationHandler<T> otherHandler = (ExpiringClassificationHandler<T>) other;
        if (otherHandler.expiry.isAfter(expiry))
            this.expiry = this.expiry.plus(Duration.between(expiry, otherHandler.expiry));
    }

    @Override
    public Prediction prediction() {
        return handler.prediction();
    }

    @Override
    public PredictionHandler newHandlerInstance() {
        return new ExpiringClassificationHandler(expiryDuration);
    }

    @Override
    public boolean shouldPrune() {
        return Instant.now().isAfter(expiry);
    }

    @Override
    public void cleanup() {

    }

    public Instant getExpiry() {
        return expiry;
    }

}
