void printarray(int *array, int start, int end, char *spacing)
{
	int i;

	for (i = start; i < end; ++i)
		printf("%d%s", array[i], spacing);

	printf("\n");
}

void printfloatarray(float *array, int start, int end, char *spacing)
{
	int i;

	for(i = start; i < end; i++)
		printf("%f%s", array[i], spacing);

	printf("\n"); 
	
}

void printtwoarray(int *array1, float *array2, int start, int end, char *spacing1, char *spacing2)
{
	int i;

	for (i = start; i < end; ++i)
		printf("%d%s%f%s", array1[i], spacing1, array2[i], spacing2);

	printf("\n");	
}

int printarraywithsum(int *array, int start, int end, char *spacing)
{
	int i;
	int sum = 0;

	for (i = start; i < end; ++i)
	{
		sum = sum + array[i];
		printf("%d%s", array[i], spacing);
	}
	printf("SUM = %d\n\n", sum);

	return sum;
}
