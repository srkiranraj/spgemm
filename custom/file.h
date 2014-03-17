void convertMTX2CSR(char *filepath, int *rowIndex, int *columnIndex, int *values, int *nzPerRow, int *rIndex, int rows, int cols, int nz)
{
	FILE *fp;
	int row_id, col_id;
	int old_row_id = 0;
	int index_pointer = 1;
	int i;

	fp = fopen(filepath, "r");
	if(fp == NULL)
	{
		perror("Error while opening the file.\n");
		exit(EXIT_FAILURE);
	}

	fscanf(fp, "%d", &rows);
	fscanf(fp, "%d", &cols);
	fscanf(fp, "%d", &nz);

	for (i = 0; i < nz; ++i)
	{
		fscanf(fp, "%d", &col_id);
		fscanf(fp, "%d", &row_id);

		values[i] = 1;
		columnIndex[i] = col_id - 1;


		if (old_row_id != row_id - 1)
		{
			rIndex[index_pointer] = index_pointer;
			rowIndex[index_pointer] = i;
			index_pointer++;
		}
		old_row_id = row_id-1;


		nzPerRow[row_id - 1] = nzPerRow[row_id - 1] + 1;
	}
	rowIndex[index_pointer] = nz;

	fclose(fp);
}
