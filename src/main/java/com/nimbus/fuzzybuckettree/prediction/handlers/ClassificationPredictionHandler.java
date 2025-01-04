package com.nimbus.fuzzybuckettree.prediction.handlers;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.nimbus.fuzzybuckettree.prediction.Prediction;
import com.nimbus.fuzzybuckettree.prediction.PredictionHandlers;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class ClassificationPredictionHandler<T> implements PredictionHandler<T> {

    @JsonProperty
    private final Map<T, Integer> labels;
    @JsonProperty
    private volatile T mostFrequentLabel;
    @JsonProperty
    private volatile int highestCount;
    private volatile int totalCount;

    @JsonCreator
    ClassificationPredictionHandler(
            @JsonProperty("labels") Map<T, Integer> labels,
            @JsonProperty("mostFrequentLabel") T mostFrequentLabel,
            @JsonProperty("highestCount") int highestCount
    ) {
        this.labels = new ConcurrentHashMap<>(labels);
        this.mostFrequentLabel = mostFrequentLabel;
        this.highestCount = highestCount;
        this.totalCount = labels.values().stream().mapToInt(Integer::intValue).sum();
    }

    public ClassificationPredictionHandler() {
        this.labels = new HashMap<>();  // Changed to regular HashMap
        this.mostFrequentLabel = null;
        this.highestCount = 0;
        this.totalCount = 0;
    }

    @Override
    public void record(T outcomeValue) {
        synchronized(this) {
            Integer count = labels.get(outcomeValue);
            int newCount = (count == null) ? 1 : count + 1;
            labels.put(outcomeValue, newCount);
            totalCount++;

            if (newCount > highestCount) {
                highestCount = newCount;
                mostFrequentLabel = outcomeValue;
            }
        }
    }

    @Override
    public Prediction prediction() {
        T currentLabel = mostFrequentLabel;
        int currentHighest = highestCount;
        int currentTotal = totalCount;

        if (currentLabel == null || currentTotal == 0) {
            return null;
        }

        // Calculate confidence without streaming
        float confidence = (float) currentHighest / currentTotal;
        return new Prediction(currentLabel, confidence);
    }

    @Override
    public ClassificationPredictionHandler<T> newHandlerInstance() {
        return new ClassificationPredictionHandler<>();
    }

    @Override
    public void merge(PredictionHandler<T> other) {
        if (!(other instanceof ClassificationPredictionHandler))
            throw new IllegalArgumentException("ClassificationPredictionHandler can only be merged with ClassificationPredictionHandler");

        ClassificationPredictionHandler<T> otherHandler = (ClassificationPredictionHandler<T>) other;

        synchronized(this) {
            otherHandler.labels.forEach((label, count) -> {
                Integer existingCount = this.labels.get(label);
                int newCount = (existingCount == null) ? count : existingCount + count;
                this.labels.put(label, newCount);

                if (newCount > this.highestCount) {
                    this.highestCount = newCount;
                    this.mostFrequentLabel = label;
                }
            });

            this.totalCount += otherHandler.totalCount;
        }
    }

}