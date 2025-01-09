package com.nimbus.fuzzybuckettree;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.nimbus.fuzzybuckettree.prediction.NodePrediction;
import com.nimbus.fuzzybuckettree.prediction.Prediction;
import com.nimbus.fuzzybuckettree.prediction.handlers.PredictionHandler;

import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * Represents a single feature value node in the tree
 * @param <V> Type of value this feature represents. String, float, etc
 * @param <T> Type of prediction value this node returns
 */
class FeatureNode<V, T> {

    static final FeatureConfig FINAL_LEAF = new FeatureConfig("leaf", FeatureValueType.FLOAT, 0, null);

    @JsonProperty
    private final boolean isRoot;
    @JsonProperty
    private final FeatureConfig feature;
    @JsonIgnore
    private transient List<FeatureConfig> allFeatures;
    @JsonIgnore
    private transient Map<String, FeatureConfig> featureCache;
    @JsonIgnore
    private transient FeatureConfig nextFeature;

    /**
     * Mapping of the hash value of the unique feature value combinations to
     * their related child nodes, if any
     */
    @JsonProperty
    private final Map<String, FeatureNode<V, T>> children;
    /**
     * Prediction handler to retrieve the prediction value at this current depth in the tree,
     * without going down any child nodes. Allows us to optionally make "summarizing" predictions
     * on parent nodes if there is deemed insufficient decisioning data at full tree depth
     */
    @JsonProperty
    private final PredictionHandler predictionHandler;

    @JsonProperty("allFeatures")
    @JsonInclude(JsonInclude.Include.NON_NULL)
    private List<FeatureConfig> getAllFeaturesIfRoot() {
        return isRoot ? allFeatures : null;
    }

    @JsonCreator
    public static <V, T> FeatureNode<V, T> deserialize(
            @JsonProperty("feature") FeatureConfig feature,
            @JsonProperty("allFeatures") List<FeatureConfig> allFeatures,
            @JsonProperty("featureCache") Map<String, FeatureConfig> featureCache,
            @JsonProperty("children") Map<String, FeatureNode<V, T>> children,
            @JsonProperty("predictionHandler") PredictionHandler<T> predictionHandler,
            @JsonProperty("isRoot") boolean isRoot) {

        FeatureNode<V, T> node = new FeatureNode<>(feature, allFeatures, featureCache, predictionHandler, isRoot);

        // If this is not the root node, allFeatures will be null
        // The root node will have populated allFeatures from deserialization
        if (children != null) {
            node.children.putAll(children);
            // Propagate allFeatures to all children
            if (allFeatures != null) {
                propagateFeatures(node, allFeatures, featureCache);
            }
        }

        return node;
    }

    // Helper method to assist avoiding re-serializing global feature configs
    private static <V, T> void propagateFeatures(FeatureNode<V, T> node, List<FeatureConfig> features, Map<String, FeatureConfig> featureCache) {
        node.allFeatures = features;
        node.featureCache = featureCache;
        for (FeatureNode<V, T> child : node.children.values()) {
            propagateFeatures(child, features, featureCache);
        }
    }

    /**
     * Construct a feature node which represents the value(s) for a particular label and their associated
     * individual rounding buckets if applicable. E.g. if a tree is setup to predict an animal,
     * this may be a feature of "barks" which could have a single value true/false. Additionally
     * features can be comprised of collective ordered values, e.g. time windows of data in which
     * their ordering is preserved
     * @param feature {@link FeatureConfig} Specifying this feature label, value count, and bucketing if applicable
     * @param predictionHandler {@link PredictionHandler} implementation. Can be a default of regression, classification
     *                                                   or custom
     */
    FeatureNode(FeatureConfig feature, List<FeatureConfig> allfeatures, Map<String, FeatureConfig> featureCache, PredictionHandler<T> predictionHandler, boolean isRoot) {
        this.feature = feature;
        this.allFeatures = allfeatures;
        this.children = new ConcurrentHashMap<>();
        this.predictionHandler = predictionHandler;
        this.isRoot = isRoot;
        this.featureCache = featureCache;
    }

    private String hash(Object[] values) {
        return Arrays.stream(values).map(v -> v.toString()).collect(Collectors.joining(":"));
    }

    private void validateFvp(FeatureValuePair fvp) {
        if (fvp.values() == null)
            throw new IllegalArgumentException("Value map for prediction missing value(s) for feature " + feature.label());
        if (fvp.label() == null || fvp.label().isEmpty() || !fvp.label().equals(feature.label()))
            throw new IllegalArgumentException("Prediction order for feature " + feature.label() + " doesnt match label " + fvp.label());
        if (fvp.values().length != feature.valueCount())
            throw new IllegalArgumentException("Prediction values for feature " + feature.label() + " must have length " + feature.valueCount() + " but had " + fvp.values().length);
        if (!this.feature.type().getArrayClass().isInstance(fvp.values())) {
            throw new IllegalArgumentException("Training values of type " +
                    fvp.values().getClass() + " must be of configured type " +
                    feature.type().getArrayClass().getName());
        }
    }

    private FeatureConfig getNextFeatureConfig(FeatureValuePair[] stack, int depth) {
        if (this.nextFeature != null)
            return this.nextFeature;

        if (depth < stack.length)
            this.nextFeature = this.featureCache.get(stack[depth].label());
        else
            this.nextFeature = FINAL_LEAF;

        return this.nextFeature;
    }

    public void update(FeatureValuePair[] featureStack, T outcome) {
        update(featureStack, 0, outcome);
    }

    private void update(FeatureValuePair[] featureStack, int depth, T outcome) {
        if (depth == featureStack.length) {
            // we are inside the final leaf node, update our records and bounce
            this.predictionHandler.record(outcome);
            return;
        }

        FeatureValuePair<V> fvp = featureStack[depth];
        try {
            this.validateFvp(fvp);
        } catch (Throwable e) {
            System.out.println("Feature stack: " + Arrays.toString(featureStack) + " at depth " + depth);
            e.printStackTrace();
            throw e;
        }

        V[] rounded = bucketValues((V[]) fvp.values());
        String hash = hash(rounded);

        this.predictionHandler.record(outcome);

        if (this.feature.label() == null) {
            if (depth < featureStack.length - 1)
                throw new RuntimeException("Hit finalized leaf node but had additional features pairs in stack");
            if (!children.isEmpty())
                throw new RuntimeException("Hit finalized leaf node but had additional children. How? Panic!");
            return;
        }

        depth++;

        FeatureNode child = this.children.get(hash);
        if (child != null) {
            child.update(featureStack, depth, outcome);
            return;
        }

        FeatureConfig featureConfig = getNextFeatureConfig(featureStack, depth);
        if (featureConfig == null)
            throw new RuntimeException("Unable to find feature in config for label " + featureStack[depth].label());

        FeatureNode newChild = new FeatureNode(featureConfig, allFeatures, featureCache, predictionHandler.newHandlerInstance(), false);
        newChild.update(featureStack, depth, outcome);

        this.children.put(hash, newChild);
    }

    /**
     * Performs the bucketing (rounding, aggregation..) as configured if enabled
     * on the provided feature value input(s)
     * @param vals Feature value(s) to apply bucketing to if enabled
     * @return The bucketed feature values
     */
    private <V> V[] bucketValues(V[] vals) {
        if (vals == null || vals.length != feature.valueCount()) {
            throw new IllegalArgumentException("Values for feature " + feature.label()
                    + " must have length " + feature.valueCount());
        }

        if (!feature.type().supportsBucketing() || feature.buckets() == null || feature.buckets().length == 0) {
            return vals;
        }

        @SuppressWarnings("unchecked")
        V[] rounded = (V[]) Array.newInstance(vals.getClass().getComponentType(), feature.valueCount());

        for (int i = 0; i < rounded.length; i++) {
            V rawVal = vals[i];
            float bucket = feature.buckets()[i];

            if (bucket == 0f) {
                rounded[i] = rawVal;
                continue;
            }

            rounded[i] = switch (this.feature.type()) {
                case FLOAT -> (V) Float.valueOf(Math.round((float) rawVal / bucket) * bucket);
                case DOUBLE -> (V) Double.valueOf(Math.round((double) rawVal / bucket) * bucket);
                case INTEGER -> (V) Integer.valueOf((int)(Math.round((int) rawVal / bucket) * bucket));
                default -> rawVal;
            };
        }

        return rounded;
    }

    public NodePrediction<T> predict(FeatureValuePair<?>[] features) {
        return predict(features, 0);
    }

    private NodePrediction<T> predict(FeatureValuePair[] features, int depth) {
        if (depth == features.length) {
            // We are in the final leaf node, make our final prediction
            Prediction<T> p = this.predictionHandler.prediction();
            return p != null ? new NodePrediction<>(p, depth) : null;
        }

        FeatureValuePair fvp = features[depth];
        this.validateFvp(fvp);

        V[] rounded = bucketValues((V[])fvp.values());
        String hash = hash(rounded);

        FeatureNode child = this.children.get(hash);
        if (child != null) {
            NodePrediction<T> p = child.predict(features, depth + 1);

            if (p != null)
                return p;
        }

        Prediction<T> p = this.predictionHandler.prediction();
        return p != null ? new NodePrediction<>(p, depth) : null;
    }

    public void nodeTrace(FeatureValuePair[] features, List<FeatureNode<?, T>> stack) {
        if (this.predict(features) != null)
            stack.add(this);
    }

    /**
     * Returns the prediction handler for this node
     */
    public PredictionHandler<T> getPredictionHandler() {
        return this.predictionHandler;
    }

    /**
     * Merges another FeatureNode into this one, combining their prediction data and child nodes
     *
     * @param other The node to merge into this one
     */
    public void merge(FeatureNode<?, T> other) {
        if (other == null)
            throw new NullPointerException("Cannot merge a null FeatureNode");
        if (!this.feature.equals(other.feature)) {
            throw new IllegalArgumentException("Cannot merge a different FeatureNode " + other.feature.label()
                    + " into " + this.feature.label());
        }

        this.predictionHandler.merge(other.predictionHandler);

        for (Map.Entry<String, ? extends FeatureNode<?, T>> entry : other.children.entrySet()) {
            String hash = entry.getKey();
            FeatureNode<?, T> otherChild = entry.getValue();

            FeatureNode<V, T> thisChild = this.children.computeIfAbsent(hash,
                    k -> new FeatureNode<>(otherChild.feature, allFeatures, featureCache,
                            predictionHandler.newHandlerInstance(), otherChild.isRoot));

            thisChild.merge(otherChild);
        }
    }

    public int leafCount() {
        if (children.isEmpty() || feature == FINAL_LEAF)
            return 1;

        return children.values()
                .stream()
                .mapToInt(FeatureNode::leafCount)
                .sum();
    }

}
