tic

connection_matrix = importdata('connection_matrix');

N = size(connection_matrix,1);
A = sparse(zeros(N,N));

nodesI = zeros(N,1);
% fill in the matrix of resistors, A
R2 = sparse(0,N);
for i=1:N
    indxConnNodes = find(connection_matrix(i,:) ~= 0);
    
    A(i,i) = - sum(connection_matrix(i,indxConnNodes));
    A(i,indxConnNodes) = connection_matrix(i,indxConnNodes);
   
    nodesForR2 = indxConnNodes(indxConnNodes > i);
    pieceToAppend = zeros(length(nodesForR2),N);
    
    pieceToAppend(:,i) = connection_matrix(i,nodesForR2)';
    for k=1:length(nodesForR2)
        pieceToAppend(k,nodesForR2(k)) = - connection_matrix(i,nodesForR2(k));
    end;
    
    % append the piece to R2
    R2 = [R2 ;pieceToAppend];
end;

R2T = sparse((R2 ~= 0)');

p = (A(1,1) - 1) / 2;
A(1,1) = 1;

%system solving
I = sparse(eye(N));
[L,U,P,Q] = lu(A);
P = sparse(P);
Q = sparse(Q);
A_inv = Q * (U \ ( L \ (P * I)));

clear L;
clear U;
clear P;
clear Q;
clear I;

parfor i=1:N
    %perturbation round 1
    namda_1 = p * A_inv(1,1) + A_inv(1,i);
    mu_1 = p^2 * A_inv(1,1) + 2 * p * A_inv(1,i) + A_inv(i,i);
    
    A_inv_ii_1 = A_inv(1,1);
    
    v_A_inv_star_u_1 = namda_1 - mu_1 * A_inv_ii_1 / (1 + namda_1);
    
    %constants
    a_1 = - 1 / (1 + v_A_inv_star_u_1);
    b_1 = - 1 / (1 + namda_1) - (mu_1 * A_inv_ii_1 / (1 + namda_1) ^ 2) / (1 + v_A_inv_star_u_1);
    c_1 = (A_inv_ii_1 / (1 + namda_1)) / (1 + v_A_inv_star_u_1);
    d_1 = (mu_1 / (1 + namda_1)) / (1 + v_A_inv_star_u_1);
    
    %perturbation round 2
    %calculate epsilon
    ep = - A(i,:);
    ep(1) = - A(i,1) - 1;
    ep(i) = 1/2 * (1 - A(i,i)); 
    
    u_2 = ep;
    
    q_1 = a_1 * p + c_1 * p^2 + b_1 * p + d_1;
    q_2 = c_1 * p + b_1;
    q_3 = a_1 + c_1 * p;
    q_4 = c_1;
    
    temp_1 = q_1 * A_inv(1,:) + q_2 * A_inv(i,:);
    temp_2 = q_3 * A_inv(1,:) + q_4 * A_inv(i,:);
    
    A_at_at_inv_v_2 = A_inv(i,:) + temp_1 * A_inv(1,i) + temp_2 * A_inv(i,i);
    A_at_at_inv_u_2 = (A_inv * u_2')' + temp_1 * (A_inv(1,:) * u_2') + temp_2 * (A_inv(i,:) * u_2');
    
    namda_2 = u_2 * A_at_at_inv_v_2';
    mu_2 = u_2 * A_at_at_inv_u_2';
    
    A_inv_ii_2 = A_at_at_inv_v_2(i);
    
    v_A_inv_star_u_2 = namda_2 - mu_2 * A_inv_ii_2 / (1 + namda_2);
    
    %constants
    a_2 = - 1 / (1 + v_A_inv_star_u_2);
    b_2 = - 1 / (1 + namda_2) - (mu_2 * A_inv_ii_2 / (1 + namda_2) ^ 2) / (1 + v_A_inv_star_u_2);
    c_2 = (A_inv_ii_2 / (1 + namda_2)) / (1 + v_A_inv_star_u_2);
    d_2 = (mu_2 / (1 + namda_2)) / (1 + v_A_inv_star_u_2);
    
    for j=i+1:N
        X = A_inv(j,:) + temp_1 * A_inv(1,j) + temp_2 * A_inv(i,j) ...
        + a_2 * A_at_at_inv_v_2(j) * A_at_at_inv_u_2 ... 
        + b_2 * A_at_at_inv_u_2(j) * A_at_at_inv_v_2 ... 
        + c_2 * A_at_at_inv_u_2(j) * A_at_at_inv_u_2 ...
        + d_2 * A_at_at_inv_v_2(j) * A_at_at_inv_v_2;
        
        i_branch = R2 * X';
        % Solve for all the currents in nodes (what we want)
        i_nodes = R2T * abs(i_branch)/2;
        % set the contribution of the end nodes to 0
        i_nodes([j i]) = 0;
        nodesI = nodesI + i_nodes;
    end
    
end

save('information_flow_result', 'nodesI', '-ascii')

toc
