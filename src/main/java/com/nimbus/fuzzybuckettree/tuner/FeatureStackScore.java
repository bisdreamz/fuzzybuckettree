package com.nimbus.fuzzybuckettree.tuner;

/**
 * Describes the state of model improvement with addition of this feature, and its score
 * at its assigned depth. Keeps a trail of model score improvements as additional features
 * are added to the list
 * @param feature
 * @param score
 */
public record FeatureStackScore(String feature, float score) {
}
