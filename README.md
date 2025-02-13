# Bipartite Graph Clustering

## Requirements
- Linux machine 
- Python3
- numPy = 1.23.3
- scikit-learn = 1.1.2
- scipy = 1.9.2
- argparse = 1.1
- munkres = 1.1.4

## Large Datasets
Generate a dataset in the format: hstid, appid, weight (separated with tabs)
e.g.
```csv
0	0	0.12178742921850087
0	1	0.006738983188443142
0	2	8.35313582480523
1	3	8.894910145883115
1	4	0.08918930084656412
1	5	0.046147818360451114
...
```


## Clustering
Remember to go into the right directory using `cd src` and then run:
```shell
$ python3 -W ignore run.py --data dataset --k 4
```
Note that "--dim 5" means that the beta parameter in the paper is set to 5*k.
The clustering results can be found in the folder "cluster".

## Citation
```
@article{YangShi23,
  author       = {Renchi Yang and
                  Jieming Shi},
  title        = {Efficient High-Quality Clustering for Large Bipartite Graphs},
  journal      = {Proceedings of the ACM on Management of Data},
  volume       = {2},
  number       = {1},
  pages        = {23:1--23:27},
  year         = {2024}
}
```
