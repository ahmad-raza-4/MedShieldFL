## medshieldfl: robust privacy protection in federated medical imaging

medshieldfl is a privacy-preserving framework for federated medical image classification with vision transformers. it focuses on improving privacy without sacrificing model accuracy in multi-institutional collaborations.

instead of adding uniform noise to all parameters, medshieldfl uses fisher information to find parameters that matter most for predictions. it applies less noise to these important parameters and more to less critical ones, adjusting noise levels during training for better convergence.

### key features
* federated vision transformer support
* adaptive noise injection based on parameter importance
* dynamic noise scheduling during training
* balanced privacy–utility trade-off
* resistance to gradient leakage

### results
tested on isic'2019 and alzehimers' disease datasets, medshieldfl achieves higher accuracy than standard differential privacy methods, maintains privacy, and converges more effectively in federated setups.
