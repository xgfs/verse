#   encoding: utf8
#   convert.py
"""Converter for three common graph formats (MATLAB sparse matrix, adjacency
list, edge list) can be found in the root directory of the project.
"""

from collections import defaultdict
from scipy.io import loadmat
from struct import pack

import click
import logging
import numpy as np


MAGIC = 'XGFS'.encode('utf8')


def mat2xgfs(filename, undirected, varname):
    mat = loadmat(filename)[varname].tocsr()
    if undirected:
        mat = mat + mat.T  # we dont care about weights in this implementation
    return mat.indptr[:-1], mat.indices


def xgfs2file(outf, indptr, indices):
    nv = indptr.size
    ne = indices.size
    logging.info('num vertices=%d; num edges=%d;', nv, ne)
    outf.write(MAGIC)
    outf.write(pack('q', nv))
    outf.write(pack('q', ne))
    outf.write(pack('%di' % nv, *indptr))
    outf.write(pack('%di' % ne, *indices))


def is_numbers_only(nodes):
    for node in nodes:
        try:
            int(node)
        except ValueError:
            return False
    return True


def list2mat(input, undirected, sep):
    nodes = set()
    with open(input, 'r') as inf:
        for line in inf:
            if line.startswith('#'):
                continue
            line = line.strip()
            if sep is None:
                splt = line.split()
            else:
                splt = line.split(sep)
            if format == "edgelist":
                if len(splt) == 3:
                    if abs(float(splt[2]) - 1) >= 1e-4:
                        raise ValueError("Weighted graphs are not supported")
                    else:
                        splt = splt[:-1]
                else:
                    raise ValueError("Incorrect graph format")
            for node in splt:
                nodes.add(node)
    number_of_nodes = len(nodes)
    isnumbers = is_numbers_only(nodes)
    logging.info('Node IDs are numbers: %s', isnumbers)
    if isnumbers:
        node2id = dict(zip(sorted(map(int, nodes)), range(number_of_nodes)))
    else:
        node2id = dict(zip(sorted(nodes), range(number_of_nodes)))
    graph = defaultdict(set)
    with open(input, 'r') as inf:
        for line in inf:
            if line.startswith('#'):
                continue
            line = line.strip()
            if sep is None:
                splt = line.split()
            else:
                splt = line.split(sep)
            if isnumbers:
                src = node2id[int(splt[0])]
            else:
                src = node2id[splt[0]]
            if format == "edgelist" and len(splt) == 3:
                splt = splt[:-1]
            for node in splt[1:]:
                if isnumbers:
                    tgt = node2id[int(node)]
                else:
                    tgt = node2id[node]
                graph[src].add(tgt)
                if undirected:
                    graph[tgt].add(src)
    indptr = np.zeros(number_of_nodes + 1, dtype=np.int32)
    indptr[0] = 0
    for i in range(number_of_nodes):
        indptr[i + 1] = indptr[i] + len(graph[i])
    number_of_edges = indptr[-1]
    indices = np.zeros(number_of_edges, dtype=np.int32)
    cur = 0
    for node in range(number_of_nodes):
        for adjv in sorted(graph[node]):
            indices[cur] = adjv
            cur += 1
    return indptr[:-1], indices


def process(format, matfile_variable_name, undirected, sep, input, output):
    if format == "mat":
        indptr, indices = mat2xgfs(input, undirected, matfile_variable_name)
    elif format in ['edgelist', 'adjlist']:
        indptr, indices = list2mat(input, undirected, sep)

    with open(output, 'wb') as fout:
        xgfs2file(fout, indptr, indices)


@click.command(help=__doc__)
@click.option('--format',
              default='edgelist',
              type=click.Choice(['mat', 'edgelist', 'adjlist']),
              help='File format of input file')
@click.option('--matfile-variable-name', default='network',
              help='variable name of adjacency matrix inside a .mat file.')
@click.option('--undirected/--directed', default=True, is_flag=True,
              help='Treat graph as undirected.')
@click.option('--sep', default=' ', help='Separator of input file')
@click.argument('input', type=click.Path())
@click.argument('output', type=click.Path())
def main(format, matfile_variable_name, undirected, sep, input, output):
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO)
    logging.info('convert graph from %s to %s', input, output)
    process(format, matfile_variable_name, undirected, sep, input, output)
    logging.info('done.')


if __name__ == "__main__":
    exit(main())
