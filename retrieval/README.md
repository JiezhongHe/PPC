A running example

```
python run_exp.py --queries ../dataset/retrieval/pcms/queries/ --input_data ../dataset/retrieval/pcms/pcms_all.200.gr --para "['128-4-1-1-1','128-4-1-1-0','128-4-1-0-1','128-4-1-0-0']" --mode vc
```

where the para means the hyperparameters of the index
the index are already generated in the directory `.index`

The hyperparameters contains four configurations of index based on different types of the features (Path and Cycle) and at various level (Vertex and Edge). Each configuration contains 128 features with size of 4

