# MarkovGNN
This is the official PyTorch-Geometric implementation of MarkovGNN paper under the title "MarkovGNN: Graph Neural Networks on Markov Diffusion". This method uses different markov graphs in different layers of the GNN.

[**PDF is available in arXiv**](https://arxiv.org/abs/2202.02470)

## System requirements
Users will need to install the following tools (CPU version).
```
PyTorch: 1.7.0
PyTorch-Geometric: 1.6.1
PyTorchSparse: 0.6.8
PyTorch Scatter: 2.0.5
PyTorch Cluster: 1.5.8
PyTorch Spline Conv: 1.2.0
NetworkX: 2.2
scikit-learn: 0.23.2
Matplotlib: 3.0.3
```


## How to run
To use `random seed` disable the seed-fixing portion in the `main.py` file. A list of sample commands to run the MarkovGCN models.
```
python main.py --edgelist datasets/input2f/email.edgelist --label datasets/input2f/email.nodes.labels --eps 0.26 --epoch 200 --alpha 0.1 --nlayers 3 --lrate 0.01 --droprate 0.3 --markov_agg

python main.py --edgelist datasets/input2f/usaairports.edgelist --label datasets/input2f/usaairports.nodes.labels --oneindexed 1 --epoch 200 --alpha 1.0 --eps 0.09 --lrate 0.01 --nlayers 4 --normrow 0 --inflate 1.5 --markov_agg

python main.py --edgelist datasets/input2f/yeast.edgelist --label datasets/input2f/yeast.nodes.labels --oneindexed 1 --onelabeled 1 --eps 0.75 --epoch 200 --inflate 1.7 --lrate 0.01 --alpha 0.8 --droprate 0.1 --nlayers 3 

python main.py --edgelist datasets/input3f/squirrel_edges.txt --label datasets/input3f/squirrel_labels.txt --feature datasets/input3f/squirrel_features.txt --epoch 200 --eps 0.05 --droprate 0.25 --markov_agg --nlayers 6 --markov_agg

python main.py --edgelist datasets/input3f/chameleon_edges.txt --label datasets/input3f/chameleon_labels.txt --feature datasets/input3f/chameleon_features.txt --epoch 200 --alpha 0.8 --nlayers 3 --eps 0.2 --inflate 1.5 --droprate 0.5 --markov_agg

python main.py --edgelist datasets/input3f/chameleon_edges.txt --label datasets/input3f/chameleon_labels.txt --feature datasets/input3f/chameleon_features.txt --epoch 200 --alpha 0.2 --nlayers 2 --eps 0.06 --inflate 1.8 --droprate 0.7 --markov_agg

python main.py --eps 0.03 --droprate 0.85 --epoch 300 --alpha 0.05 --nlayers 2 --lrate 0.005 --inflate 1.8 --markov_agg

python main.py --eps 0.03 --droprate 0.85 --epoch 300 --alpha 0.05 --nlayers 2 --lrate 0.001 --inflate 3.5 --markov_agg --dataset Citeseer

python main.py --edgelist datasets/input3f/actor_edges.txt --label datasets/input3f/actor_labels.txt --feature datasets/input3f/actor_features.txt --epoch 200  --alpha 0.4 --markov_agg --nlayers 4

python main.py --edgelist datasets/input3f/actor_edges.txt --label datasets/input3f/actor_labels.txt --feature datasets/input3f/actor_features.txt --epoch 200  --alpha 0.2 --markov_agg --nlayers 3 --eps 0.3
```
To compare the results with respect to vanilla GCN, use the argument `--use_gcn` in the command line.

## Parameters
There are several options to run the method which are outlined in the `main.py` file.
```
--markov_dense -> markov process uses dense matrix multiplication (sparse matrix multiplicaiton is the default option)
--markov_agg -> i-th layer uses a markov matrix from i-th iteration, this option with higher threshold will produce better runtime
--use_gcn -> run the vanilla GCN model.
  e.g., $ python main.py --edgelist datasets/input3f/actor_edges.txt --label datasets/input3f/actor_labels.txt --feature datasets/input3f/actor_features.txt --epoch 200  --use_gcn

```

## Citation
If you find this repository helpful, please cite the following paper:
```
@article{rahman2022markovgnn,
  title={{MarkovGNN: Graph} Neural Networks on Markov Diffusion},
  author={Rahman, Md. Khaledur and Agrawal, Abhigya and Azad, Ariful},
  booktitle={arXiv preprint arXiv:2202.02470},
  year={2022}
}
```

## Contact
Please create an issue if you face any problem to run this method. Don't hesitate to contact the following person if you have any questions: Md. Khaledur Rahman (`morahma@iu.edu`).
