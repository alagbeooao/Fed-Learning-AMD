# Federated learning for abnormal motion detection from skeletons

Code for an extended version of my MSc thesis, in preparation for journal submission.

Cameras could help care-home staff notice risky behaviour such as falls or attempts to leave, but video of residents is sensitive and held at separate sites. This study tests whether an abnormal-motion detector can be trained across sites with federated learning, without sharing video, and what differential privacy costs on top.

Autoencoders learn normal movement from OpenPose skeletons (9 normal and 11 abnormal activities acted out in a care-home setting) and flag windows they reconstruct badly.

## What the notebook runs

All 315 experiments: centralised, local-only, FedAvg, FedProx and DP-SGD training of LSTM, 1D-CNN and Transformer autoencoders, across IID, quantity-skewed and activity-skewed client partitions, with PCA and a no-training score as baselines. It then builds every table and figure and checks the headline numbers.

Two evaluation shortcuts are tested explicitly: leaving undetected keypoints in the data as coordinates, and dropping difficult recordings from the test set. On its own, the share of missing keypoints in a window separates normal from abnormal with AUROC 0.73 on all test videos, and 0.87 once five difficult recordings are dropped.

## Results

| Setting | AUROC |
|---|---|
| 1D-CNN autoencoder, centralised | 0.863 |
| Transformer autoencoder, centralised | 0.844 |
| LSTM autoencoder, centralised | 0.833 |
| PCA reconstruction | 0.847 |
| 1D-CNN, FedAvg, IID clients | 0.863 |
| 1D-CNN, FedAvg, clients with disjoint activities | 0.853 |
| Worst client in that setting | 0.652 |

FedAvg stayed within 0.014 AUROC of centralised training in every setting. Differentially private training did not learn at any budget tested, because the dataset is too small for the batch sizes DP-SGD needs (ε = 34.6 at noise multiplier 1 for FedAvg on IID clients).

## Running it

Use a Colab CPU runtime; the models are small and a GPU does not help. Put the keypoint data at `My Drive/Fed-Pro/Data4.zip` and choose Run all. It takes 1.5 to 2.5 hours on a free runtime. Results are saved to Drive after every run, so after a disconnect Run all skips finished runs.

The skeleton data is not included in this repository.
