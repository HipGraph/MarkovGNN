# MarkovGNN
MarkovGNN: Graph Neural Networks using Markov Diffusion. This repository is only for WWW2022 submission.

## System requirements
Users will need to install the following tools (CPU version).
```
PyTorch: 1.7.0
PyTorch-Geometric 1.6.1
PyTorchSparse 0.6.8
PyTorch Scatter 2.0.5
PyTorch Cluster 1.5.8
PyTorch Spline Conv 1.2.0
NetworkX: 2.2
scikit-learn: 0.23.2
Matplotlib: 3.0.3
```

## How to run
A list of sample commands to run the MarkovGCN models.
```
python main.py --edgelist datasets/input2f/email.edgelist --label datasets/input2f/email.nodes.labels --eps 0.005 --epoch 200 --alpha 0.1 --nlayers 3

python main.py --edgelist datasets/input2f/usaairports.edgelist --label datasets/input2f/usaairports.nodes.labels --oneindexed 1 --epoch 200 --alpha 1.0 --eps 0.09 --lrate 0.01 --nlayers 4 --normrow 0 --inflate 1.5 --markov_agg

python main.py --edgelist datasets/input2f/yeast.edgelist --label datasets/input2f/yeast.nodes.labels --oneindexed 1 --onelabeled 1 --eps 0.25 --epoch 200 --inflate 1.5 --lrate 0.05 --alpha 0.7 --markov_agg --nlayers 3

python main.py --edgelist datasets/input3f/chameleon_edges.txt --label datasets/input3f/chameleon_labels.txt --feature datasets/input3f/chameleon_features.txt --epoch 200 --alpha 0.2 --nlayers 2 --eps 0.06 --inflate 1.8 --droprate 0.7 --markov_agg

python main.py --edgelist datasets/input3f/squirrel_edges.txt --label datasets/input3f/squirrel_labels.txt --feature datasets/input3f/squirrel_features.txt --epoch 200 --eps 0.05 --droprate 0.25 --markov_agg --nlayers 6 --markov_agg

python main.py --eps 0.03 --droprate 0.85 --epoch 300 --alpha 0.05 --nlayers 2 --lrate 0.005 --inflate 1.8 --markov_agg

python main.py --eps 0.03 --droprate 0.85 --epoch 300 --alpha 0.05 --nlayers 2 --lrate 0.001 --inflate 3.5 --markov_agg --dataset Citeseer

python main.py --edgelist datasets/input3f/actor_edges.txt --label datasets/input3f/actor_labels.txt --feature datasets/input3f/actor_features.txt --epoch 200  --alpha 0.4 --markov_agg --nlayers 4

python main.py --edgelist datasets/input3f/actor_edges.txt --label datasets/input3f/actor_labels.txt --feature datasets/input3f/actor_features.txt --epoch 200  --alpha 0.8 --markov_agg --nlayers 5
```

## Parameters
There are several options to run the method which are outlined in the main.py file.
```
--markov_dense -> markov process uses dense matrix multiplication (sparse matrix multiplicaiton is the default option)
--markov_agg -> i-th layer uses a markov matrix from i-th iteration, this option with higher threshold will produce better runtime
--use_gcn -> use vanilla GCN model
```

Please create an issue if you face any problem to run this method. We hope to respond anonymously.
