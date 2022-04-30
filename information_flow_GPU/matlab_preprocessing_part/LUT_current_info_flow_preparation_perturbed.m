connection_matrix = importdata('connection_matrix');

N = size(connection_matrix,1);
A = sparse(zeros(N,N));
A_diag = zeros(N,1);

I = eye(N);
anti_I = zeros(N,N);

for i=1:N
	if i ~= N+1-i
		anti_I(i,N+1-i) = 1;
	end
end

connection_matrix = connection_matrix + 0.001 * anti_I;

clear anti_I;

%fill in the matrix of resistors, A
for i=1:N
    A_diag(i) = - sum(connection_matrix(i,:));
end;

A = sparse(connection_matrix + diag(A_diag));

p = - A(1,1) / 2;
A(1,1) = 2 * A(1,1); %keep diagonally dominance

cond_number = cond(A)

tic
%system solving

A_inv = full(inv(A));

toc

result = double(A * A_inv);
save('Resulting_Matrix','result','-ascii');
residual = double(norm(I - result));
save('Residual_Norm','residual','-ascii');

clear I;

% column major conversion
A_inv = A_inv(:)';

fileID = fopen('chrom_1_initial_inverse_matrix.bin','w');
fwrite(fileID, A_inv, 'double');
fclose(fileID);

A_diag = diag(full(A))';

fileID = fopen('chrom_1_initial_matrix_diag.bin','w');
fwrite(fileID, A_diag, 'double');
fclose(fileID);

nonzero_rows = [];
nonzero_cols = [];
nonzero_vals = [];
nonzero_spacing = [];

[nonzero_cols, nonzero_rows, nonzero_vals] = find(connection_matrix);

nonzero_rows = nonzero_rows - 1;
nonzero_cols = nonzero_cols - 1;

nonzero_rows = nonzero_rows';
nonzero_cols = nonzero_cols';
nonzero_vals = nonzero_vals';

size(connection_matrix)

length(nonzero_rows)

nz_counter = 0;
nonzero_spacing = [nonzero_spacing, nz_counter];

for i=1:N
	nz_counter = nz_counter + nnz(connection_matrix(i,:));
	nonzero_spacing = [nonzero_spacing, nz_counter];
end

max_space = 0;

for i=1:N
	max_space = max(nonzero_spacing(i+1) - nonzero_spacing(i), max_space);
end

max_space

loop_num = ceil(log2(max_space))
LUT = [];
LUT_positions = [];
offset = 1;

LUT_positions = [LUT_positions, 0];
for i = 1:loop_num
	for j = 1:(length(nonzero_spacing)-1)
		head = nonzero_spacing(j);
		tail = nonzero_spacing(j+1);

		LUT = [LUT, head : 2*offset : (tail-1)];

		if LUT(end) + offset >= tail
			LUT(end) = [];
		end
	end

	offset = offset * 2;
	LUT_positions = [LUT_positions, length(LUT)];
end

%LUT size
LUT_size = length(LUT)

%save files
fileID = fopen('chrom_1_occupied_row_indices.bin','w');
fwrite(fileID, nonzero_rows, 'int');
fclose(fileID);

fileID = fopen('chrom_1_occupied_col_indices.bin','w');
fwrite(fileID, nonzero_cols, 'int');
fclose(fileID);

fileID = fopen('chrom_1_occupied_values.bin','w');
fwrite(fileID, nonzero_vals, 'double');
fclose(fileID);

fileID = fopen('chrom_1_occupied_spacings.bin','w');
fwrite(fileID, nonzero_spacing, 'int');
fclose(fileID);

fileID = fopen('chrom_1_LUT.bin','w');
fwrite(fileID, LUT, 'int');
fclose(fileID);

fileID = fopen('chrom_1_LUT_positions.bin','w');
fwrite(fileID, LUT_positions, 'int');
fclose(fileID);

