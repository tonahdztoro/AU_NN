In this folder can be found the same datasets as in the IW_classification/Datasets/ parent folder which are in a json structure that can be a bit difficult to interpret.
However, the difference lies that here the datasets for each subject are reestructured in a DataFrame structure to be easily interpetable.

The reason of why the first ones were saved in that way was for them to properly run with the AU_NN.py code.
Now, if it is needed to run the datasets with other models, the use the ones in DataFrame instead.

The structure goes this ay:
1. Each S(n)_dfs.pkl file contains all the data for the specific subject n (there are 27 subjects).
2. In each S(n)_dfs.pkl file it is contained a list of 5 elements (let's calle them w(m)). Each element w(m) of the list represents the data of each of the 5 words (up, down, left, right, select).
3. Each element w(m) is a list of 30 elements (let's call them df(l)). Each one of those elements represent one repetition of the word as described in the paper.
4. Each element df(l) is a DataFrame of size Rx(F+1), where R represents the
