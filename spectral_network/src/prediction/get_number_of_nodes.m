function n_nodes = get_number_of_nodes(tree)
% Calculates number of nodes in a tree

if tree.bLeaf
    n_nodes = 1;
else
    n_nodes_left = get_number_of_nodes(tree.lessthanChild);
    n_nodes_right = get_number_of_nodes(tree.greaterthanChild);
    n_nodes = n_nodes_left+n_nodes_right+1;
end