package com.nimbus.fuzzybuckettree.prediction;

import com.nimbus.fuzzybuckettree.prediction.handlers.ClassificationPredictionHandler;
import com.nimbus.fuzzybuckettree.prediction.handlers.RegressionPredictionHandler;

public class PredictionHandlers {

    public static <T> ClassificationPredictionHandler<T> classification() {
        return new ClassificationPredictionHandler<T>();
    }

    public static RegressionPredictionHandler regression() {
        return new RegressionPredictionHandler();
    }

}
