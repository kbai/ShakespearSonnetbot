from graphviz import Digraph
import numpy as np

f = Digraph('Markov_Chain_Diagram', filename='Markov', format = 'png')
f.body.extend(['rankdir=LR', 'size="10, 5"'])

A = np.array([[1, 2, 3, 4, 0, 6,7],\
		[1, 2, 3, 4, 0, 6, 1],\
		[1, 2, 3, 4, 0, 6, 1],\
		[1, 2, 3, 4, 0, 6, 1],\
		[1, 2, 3, 4, 0, 6, 1],\
		[0, 0, 0, 0, 0, 0, 1],\
		[0, 0, 0, 0, 0, 0, 1]\
	])
f.attr('node', shape='doublecircle')
f.node('START')
f.node('END')

DIM = A.shape[0] - 2

f.attr('node', shape='circle')

for it in range(DIM):
	f.edge('START', '{0}'.format(it), label= str(A[-2, it]))

for it in range(DIM):
	f.edge('{0}'.format(it), 'END', label= str(A[it, -1]))

for outer_it in range(DIM):
	for inner_it in range(DIM):
		f.edge('{0}'.format(outer_it), '{0}'.format(inner_it), label= str(A[outer_it,inner_it]))

f.view()
