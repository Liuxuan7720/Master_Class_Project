### Running

To run the optimizer, simply use

```shell
python train.py -s <path to COLMAP or NeRF Synthetic dataset>
```

### Evaluation
```shell
python train.py -s <path to COLMAP or NeRF Synthetic dataset> --eval # Train with train/test split
python render.py -m <path to trained model> # Generate renderings
python metrics.py -m <path to trained model> # Compute error metrics on renderings
```

### Running the Real-Time Viewer

```shell
./<SIBR install dir>/bin/SIBR_gaussianViewer_app -m <path to trained model>
```

### Show



https://liuxuan7720.github.io/Master_Class_Project/project_CV.mp4


