# VERSE: Versatile Graph Embeddings from Similarity Measures

This repository provides a reference implementation and data of VERSE, as described in the paper. 

## Installation and usage

For C++ executables:

    cd src && make;

should be enough on most platforms. If you need to change the default compiler (i.e. to Intel), use:

    make CXX=icpc

VERSE is able to encompass diverse similarity measures under its model. For performance reasons, we have implemented three different similarities separately.

Use the command

    verse -input data/karate.bcsr -output karate.bin -dim 128 -alpha 0.85 -threads 4 -nsamples 3

to run the default version (that corresponds to PPR similarity) with 128 embedding dimension, PPR alpha 0.85, using 3 negative samples.

## Graph file format

This implementation uses a custom graph format, namely binary [compressed sparse row](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_.28CSR.2C_CRS_or_Yale_format.29) (BCSR) format for efficiency and reduced memory usage. Converter for three common graph formats (MATLAB sparse matrix, adjacency list, edge list) can be found in the ``python`` directory of the project. Usage:

```
$ convert-bcsr --help
Usage: convert-bcsr [OPTIONS] INPUT OUTPUT

  Converter for three common graph formats (MATLAB sparse matrix, adjacency
  list, edge list) can be found in the root directory of the project.

Options:
  --format [mat|edgelist|adjlist]
                                  File format of input file
  --matfile-variable-name TEXT    variable name of adjacency matrix inside a
                                  .mat file.
  --undirected / --directed       Treat graph as undirected.
  --sep TEXT                      Separator of input file
  --help                          Show this message and exit.
```

1. ``--format adjlist`` for an adjacency list, e.g:

        1 2 3 4 5 6 7 8 9 11 12 13 14 18 20 22 32
        2 1 3 4 8 14 18 20 22 31
        3 1 2 4 8 9 10 14 28 29 33
        ...

1. ``--format edgelist`` for an edge list, e.g:

        1 2
        1 3
        1 4
        ...

1. ``--format mat`` for a Matlab MAT file containing an adjacency matrix
        (note, you must also specify the variable name of the adjacency matrix ``--matfile-variable-name``)

## Citation

If you use the code or the datasets, please consider citing out .

    @article{tsitsulin2018,
        title={VERSE: Versatile Graph Embeddings from Similarity Measures},
        author={Tsitsulin, Anton and Mottin, Davide and Karras, Panagiotis and M{\"u}ller, Emmanuel},
        booktitle={Proceedings of The Web Conference 2018},
        year={2018},
        organization={ACM}
    }

## Full VERSE

This repository only contains the code of the scalable VERSE version. In order to obtain fVERSE, please use [this link](https://github.com/xgfs/fverse). Note, however, that fVERSE code is not well documented and/or supported.

## Contact

`echo "%7=87.=<2=<>527@192.()" | tr '#-)/->' '_-|'`