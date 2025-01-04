# FuzzyBucketTree

FuzzyBucketTree is a flexible and efficient machine learning tree structure designed for pattern recognition and prediction tasks. It combines the simplicity of decision trees with powerful fuzzy matching capabilities, making it particularly well-suited for real-world data with approximate patterns.

## Key Features

* **Smart Value Aggregation**
  * Automatic "bucketing" of numerical values enables improved approximate pattern recognition
  * Values are intelligently grouped to identify meaningful patterns while reducing noise
  * Configurable bucket sizes per feature for fine-tuned control over pattern matching

* **Flexible Feature Representation**
  * Features can consist of single values or ordered sequences
  * Support for time-series data through ordered value sequences
  * Each feature can be configured independently with its own value count and bucketing strategy

* **Automated Optimization**
  * Built-in auto-tuning capability explores feature order permutations
  * Automatic bucket size optimization for numerical features
  * Finds optimal tree structure without manual intervention

* **Online Learning**
  * Supports continuous learning from new data
  * Model updates are thread-safe and efficient
  * No separate training phase required - learn as you go

* **Versatile Prediction Handlers**
  * Built-in support for classification and regression tasks
  * Extensible prediction handler interface for custom logic
  * Implement custom criteria for predictions (e.g., minimum sample size)
  * Create complex prediction strategies for unique use cases

* **Multi-Domain Support**
  * Separate domains for different types of predictions
  * Independent learning and optimization per domain
  * Clean separation of concerns for different prediction tasks
  * Efficient organization of related pattern recognition tasks

* **Model Management**
  * Simple model persistence with save/load functionality
  * Ability to merge models trained on different datasets
  * Easy model sharing and deployment

## Quick Start

```java
// Create a new tree with specified features
List<FeatureConfig> features = Arrays.asList(
    new FeatureConfig("temperature", 1, new float[]{0.5f}),
    new FeatureConfig("humidity", 1, new float[]{5f})
);
FuzzyBucketTree<String> tree = new FuzzyBucketTree<>(features, new ClassificationPredictionHandler<>());

// Train the model
Map<String, float[]> featureValues = new HashMap<>();
featureValues.put("temperature", new float[]{24.5f});
featureValues.put("humidity", new float[]{65f});
tree.train(featureValues, "sunny");

// Make predictions
NodePrediction<String> prediction = tree.predict(featureValues);
```

## Multi-Domain Usage

```java
// Create a multi-domain tree
MultiTree<String> multiTree = new MultiTree<>();

// Add domain-specific trees
multiTree.addDomain("weather", weatherTree);
multiTree.addDomain("traffic", trafficTree);

// Train domain-specific predictions
multiTree.train("weather", weatherFeatures, "sunny");
multiTree.train("traffic", trafficFeatures, "congested");
```

## Model Persistence

```java
// Save model
tree.saveModel("weather_model.fbt");

// Load model
FuzzyBucketTree<String> loadedTree = FuzzyBucketTree.loadModel("weather_model.fbt");

// Merge models
tree.merge(anotherTree);
```

## Custom Prediction Handlers

```java
public class CustomPredictionHandler<T> implements PredictionHandler<T> {
    @Override
    public void record(T outcomeValue) {
        // Custom recording logic
    }

    @Override
    public Prediction<T> prediction() {
        // Custom prediction logic
    }
}
```

## Auto-Tuning

FuzzyBucketTree includes a powerful auto-tuning capability that optimizes both feature ordering and bucket configurations to maximize prediction accuracy.

### Basic Auto-Tuning

```java
// Define feature options for tuning
List<FeatureBucketOptions> featureOptions = Arrays.asList(
    new FeatureBucketOptions("temperature", 1, new float[]{0.5f, 1.0f, 2.0f}),  // Try different bucket sizes
    new FeatureBucketOptions("humidity", 1, null)  // No bucketing for this feature
);

// Create tuner with classification prediction
FuzzyBucketTuner<String> tuner = new FuzzyBucketTuner<>(
    featureOptions,
    PredictionHandlers.classification(),
    new ClassificationReporter<>()
);

// Train and tune using separate training/validation sets
List<TrainingEntry<String>> training = loadTrainingData();
List<TrainingEntry<String>> validation = loadValidationData();
TunerResult<String> result = tuner.train(training, validation);

// Access tuning results
System.out.println("Final accuracy: " + result.getTotalAccuracy());
result.getAccuracyReports().forEach((label, report) -> {
    System.out.println(String.format(
        "%s: %.2f%% accuracy (%d samples)",
        label,
        report.calcWeightedAccuracy() * 100,
        report.getSamples()
    ));
});
```

The auto-tuner will:
- Explore different feature orderings to find optimal tree structure
- Test different bucket configuration combinations for each feature permutation
- Evaluate combinations using validation data
- Report detailed accuracy metrics per prediction label
- Produce an optimized tree configuration for your use case
- Stop early if a perfect result is found, or if no improvement is made (e.g. features are order agnositc)

## Contributing
Holler
