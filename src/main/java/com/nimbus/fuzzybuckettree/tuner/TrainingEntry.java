package com.nimbus.fuzzybuckettree.tuner;

import java.util.Map;

public record TrainingEntry<T> (Map<String, float[]> features, T outcome) {
}
