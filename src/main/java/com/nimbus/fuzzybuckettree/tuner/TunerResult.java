package com.nimbus.fuzzybuckettree.tuner;

import com.nimbus.fuzzybuckettree.FeatureConfig;
import com.nimbus.fuzzybuckettree.FuzzyBucketTree;
import com.nimbus.fuzzybuckettree.tuner.reporters.AccuracyReporter;
import com.nimbus.fuzzybuckettree.tuner.reporters.PredictionReport;

import java.util.List;
import java.util.Map;

public class TunerResult<T> {

    private final List<FeatureConfig> featureConfigs;
    private final AccuracyReporter<T> reporter;
    private final FuzzyBucketTree<T> tree;

    public TunerResult(List<FeatureConfig> featureConfigs, AccuracyReporter<T> reporter, FuzzyBucketTree<T> tree) {
        this.featureConfigs = featureConfigs;
        this.reporter = reporter;
        this.tree = tree;
    }

    public List<FeatureConfig> getFeatureConfigs() {
        return featureConfigs;
    }

    public float getTotalAccuracy() {
        return reporter.getTotalAccuracy();
    }

    public Map<T, PredictionReport> getAccuracyReports() {
        return reporter.getAccuracyReports();
    }

    public FuzzyBucketTree<T> getTree() {
        return tree;
    }
}
