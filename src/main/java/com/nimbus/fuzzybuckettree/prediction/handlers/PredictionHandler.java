package com.nimbus.fuzzybuckettree.prediction.handlers;


import com.nimbus.fuzzybuckettree.prediction.Prediction;

/**
 * Responsible for tracking samples and outcome values for a given node,
 * and calculating the corresponding prediction outcomes for that node
 */
public interface PredictionHandler<T> {

    /**
     * Record a known prediction outcome to learn in an online fashion
     * @param outcomeValue
     */
    public void record(T outcomeValue);

    /**
     * For tree merging, merge in results from a prediction handler from another tree
     * to so this prediction handler now knows the sum of both tree observations
     * @param other
     */
    public void merge(PredictionHandler<T> other);

    public Prediction prediction();

    public PredictionHandler<T> newHandlerInstance();

    /**
     * @return True if this handler determines the data is stale and its associated node
     * should be pruned
     */
    public boolean shouldPrune();

    /**
     * Called if this handler's node is being pruned, do any clean up work if applicable
     */
    public void cleanup();

}
