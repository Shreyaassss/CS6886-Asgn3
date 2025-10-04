
# Systems Engineering for Deep Learning – Model Compression

This is the source code for the deep learning model compression assignment, where **Deep Compression** techniques are applied on the **MobileNetV2** model. 

---

## Replicating the Results

```bash
cd <asgn directory>
```



---


```bash
python3 ./train_baseline.py
```

To change tunable arguments:

```bash
python3 ./train_baseline.py --help
```

This will create the NN model and provide the baseline accuracy of the network.
No compression tricks have been performed yet.

Save the checkpoint for baseline in a convenient location. This will be used as an argument in the next instruction.

---

## Compression Stages

### 1. Pruning

```bash
python3 ./pruning.py --checkpoint <Best-Baseline-Checkpoint>
```

To change tunable arguments:

```bash
python3 ./pruning.py --help
```

Save the best pruned model checkpoint at a convenient location.

---

### 2. Quantization

```bash
python3 ./quantization.py --checkpoint <Best-Pruned-Checkpoint>
```

To change tunable arguments:

```bash
python3 ./quantization.py --help
```
