%this code is downloaded from the original paper website http://people.csail.mit.edu/patrycja/ with deletion of redundant code for speeding up and additional codes time the calculations 
connectMatrix = importdata('connection_matrix');

if ~exist('condOrRes','var')
    condOrRes = 1;
end;

N = size(connectMatrix,1);
A = sparse(zeros(N,N));
nodesI = zeros(N,1);
% Step 1: fill in the matrix of resistors, A
% Also, set up matrix R2 needed for Step 6, also sparse
% it is used to solve for the branch (NOT node) currents (eg. between nodes
% i and j), size M by N where N is number of nodes, M is number of branches
R2 = sparse(0,N);
for i=1:N
    indxConnNodes = find(connectMatrix(i,:) ~= 0);
    % add their values up
    if (condOrRes == 1) % conductance
        A(i,i) = - sum(connectMatrix(i,indxConnNodes));
    else % resistance
        A(i,i) = - sum(1 / connectMatrix(i,indxConnNodes));
    end;
    % fill in the off diagonals, only the one for the row we are working
    % with (the upper one)
    if (condOrRes == 1) % conductance
        A(i,indxConnNodes) = connectMatrix(i,indxConnNodes);
    else % resistance
        A(i,indxConnNodes) = 1 / connectMatrix(i,indxConnNodes);
    end;
    
    % setup R2 (used later on)
    % for each of the branches, we need to put a row into the matrix with 
    % conductance/resistance on it
    nodesForR2 = indxConnNodes(indxConnNodes > i);
    pieceToAppend = zeros(length(nodesForR2),N);
    if (condOrRes == 1)
        pieceToAppend(:,i) = connectMatrix(i,nodesForR2)';
        for k=1:length(nodesForR2)
            pieceToAppend(k,nodesForR2(k)) = - connectMatrix(i,nodesForR2(k));
        end;
    else
        pieceToAppend(:,i) = 1 / connectMatrix(i,nodesForR2)';
        for k=1:length(nodesForR2)
            pieceToAppend(k,nodesForR2(k)) = - 1/ connectMatrix(i,nodesForR2(k));
        end;
    end;
    % append the piece to R2
    R2 = [R2 ;pieceToAppend];
end;
% matrix A and R2 only need to be set up once, done
% also form R2T (transpose of R2, with 1s where nonzero entries are), used
% in step 7

R2T = sparse((R2 ~= 0)');

% Step 2: Iterate over the ground node i_g=1:N, force the ground node to 0
% by setting the appropriate row of A to all zeros except 1 at the i_g
% position.
count = 0;
for ig=1:N
    tic
    ig

    % make a copy of A matrix, and force one node to ground
    Ag = sparse(A);
    Ag(ig,:) = zeros(1,N);
    Ag(ig,ig) = 1;
    % this Ag matrix stays the same for the next loop, thus we can do LU
    % decomposition on it once per ground node
    
    % Step 3: LU decomposition of matrix Ag
    [L,U,P, Q] = lu(Ag);
    
    % Step 4: within the ground node loop, iterate over the source node, and set the
    % RHS to 1 for that node

    % i_vec is net vector of currents into the node,
    % we have A * phi = i_vec, where phi is a vector to solve for,
    % phi is a vector of node voltages
    i_vec = zeros(N,1);
    for is = ig+1:N % 
        i_vec(is) = 1; % set this value to the current source of 1 Amp
        
        % Step 5: solve for phi - node voltages
        phi = Q*(U\  (L\  (P * i_vec)));
      
        % Step 6: Solve for current value flowing through each node        
        % R2 * phi = i_branch (i_branch is the current in each branch, size
        % m by 1, m is number of branches
        i_branch = R2 * phi;
        % Step 7: Solve for all the currents in nodes (what we want)
        i_nodes = R2T * abs(i_branch)/2;
        % set the contribution of the end nodes to 0
        i_nodes([is ig]) = 0;
        nodesI = nodesI + i_nodes;
        
        count = count + 1;
        % reset the vector to zeros
        i_vec(is) = 0;
    end;
    
    toc
        
end;

if (count ~= 0)
    nodesAverageI = nodesI / count;
else 
    nodesAverageI = 0;
end;

save('information_flow_result', 'nodesI', '-ascii')


