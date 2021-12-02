#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <mpi.h>
#include <string.h>

#define accuracy 0.00001

//у каждого процесса актуальная матрица
//m - столбцы, n - строки


int ProcRank = 0;
int ProcNum = 0;
int real_m = 0,  from_m = 0;
int * recvcounts_for_m = NULL, * displs_for_m = NULL;
int n,m;

double * get_matrix(int n,int m, FILE* myfile)
{
    double * matrix = calloc(n * m,sizeof(double));

    for (int i=0;i<m*n;i++)
        fscanf(myfile,"%lf",matrix+i);

    return matrix;
}

// ищет первый ненулевой элемент в строке
int find_major_element(double * matrix, int m)
{
    for (int i=0;i<m;i++)
    {
        if (matrix[i]>=accuracy || matrix[i]<=-accuracy)
        {
            return i+1;
        }
    }
    return 0;
}


//ведущие элементы всех строк 
int * find_major_elements_array(double * matrix,int n, int m)
{
    int * array = calloc(n,sizeof(int));
    int * buf_send = calloc(n,sizeof(int));

    int real_n = n / ProcNum;

    int buf_size = n % ProcNum;

    int recvcounts[ProcNum];
    int displs[ProcNum];

    for (int i=0; i<ProcNum; i++)
    {

        if (i < buf_size)
        {
            recvcounts[i] = real_n + 1;
            displs[i] = (i * real_n + i);
        }
        else
        {
            recvcounts[i] = real_n;
            displs[i] = (i * real_n + buf_size);
        }

    }

    int from_n = displs[ProcRank];
    real_n = recvcounts[ProcRank];

    for (int i=from_n;i<from_n + recvcounts[ProcRank];i++)
    {
        array[i] = find_major_element(matrix + i*m, m);
    }

    memcpy(buf_send, array+from_n, real_n * sizeof(int));
    MPI_Allgatherv(buf_send,real_n,MPI_INT,array,recvcounts,displs,MPI_INT,MPI_COMM_WORLD);

    free(buf_send);

    return array;
}


//меняем строки местами, если главный элемент имеет более маленький индекс
void swap_strings(double * matrix,int i,int m)
{
    double buf;

    double * buf_send = calloc(n*m, sizeof(double));

    for (int j = from_m; j < from_m + real_m; j++)
    {
        buf = *(matrix+j);
        *(matrix+j) = *(matrix+i*m+j);
        *(matrix+i*m+j) = buf;
    }

    memcpy(buf_send, matrix + from_m, real_m * sizeof(double));
    MPI_Allgatherv(buf_send, real_m, MPI_DOUBLE, matrix, recvcounts_for_m, displs_for_m, MPI_DOUBLE, MPI_COMM_WORLD);

    memcpy(buf_send, matrix + i*m + from_m, real_m * sizeof(double));
    MPI_Allgatherv(buf_send, real_m, MPI_DOUBLE, matrix + i*m, recvcounts_for_m, displs_for_m, MPI_DOUBLE, MPI_COMM_WORLD);

    free(buf_send);
}

//подготавливаем ведуший элемент (строка, у которой самый левый ненулевой элемент выносится наверх матрицы)
void step_one(double * matrix, int n, int m, int * array)
{
    int min = array[0];

    for (int i=1;i<n;i++)
    {
        if (array[i] && min>array[i])
        {
            swap_strings(matrix,i,m);
            min = array[i];
            array[i] = array[0];
            array[0] = min;
        }
    }
}

void swap_string_with_zero(double * matrix, int n1, int n2, int m)
{
    if (n1==n2)
        return;

    double * buf_send = calloc(n*m, sizeof(double));

    for (int i=from_m;i<from_m+real_m;i++)
    {
        *(matrix+n1*m + i) = *(matrix+n2*m + i);
        *(matrix+n2*m + i) = 0;
    }

    memcpy(buf_send, matrix + n1*m + from_m, real_m * sizeof(double));
    MPI_Allgatherv(buf_send, real_m, MPI_DOUBLE, matrix + n1*m, recvcounts_for_m, displs_for_m, MPI_DOUBLE, MPI_COMM_WORLD);

    memcpy(buf_send, matrix + n2*m + from_m, real_m * sizeof(double));
    MPI_Allgatherv(buf_send, real_m, MPI_DOUBLE, matrix + n2*m, recvcounts_for_m, displs_for_m, MPI_DOUBLE, MPI_COMM_WORLD);
}

int zero_string_to_end(double * matrix,int n, int m,int * array)
{
    int amount = 0;
    int i = 0;
    while (i<=n-amount-1)
    {
        if (array[i]==0)
        {
            swap_string_with_zero(matrix,i,n-amount-1,m);
            array[i]=array[n-amount-1];
            array[n-amount-1] = 0;
            amount++;
        }
        else
            i++;
    }
    return amount;
}

void output_matrix(double * matrix, int n, int m, FILE * output_file)
{
    for (int i=0;i<n;i++)
    {
        for (int j=0;j<m;j++)
            fprintf(output_file,"%lf ",*(matrix+i*m+j));
        fprintf(output_file,"\n");
    }
    fprintf(output_file,"\n");
}


int to_trapec_matrix(double * matrix, int n, int m)
{
    int * major_elements_array = find_major_elements_array(matrix,n,m);

    double buf;

    double * buf_matrix = matrix;
    double * buf_send = calloc(n * m,sizeof(double));

    int real_n, from_n;
    int recvcounts_for_n[ProcNum];
    int displs_for_n[ProcNum];
    int buf_size = 0;

    n -= zero_string_to_end(buf_matrix,n,m,major_elements_array);
    // все нулевые строки в конец


    for (int i=0;i<n-1;i++)
    {
        step_one(buf_matrix, n-i, m, major_elements_array);

        real_n = (n - i - 1) / ProcNum;
        buf_size = (n - i - 1) % ProcNum;
        for (int i=0; i<ProcNum; i++)
        {
            if (i < buf_size)
            {
            recvcounts_for_n[i] = (real_n + 1) * m;
            displs_for_n[i] = i * (real_n + 1) * m + m;
            }
            else
            {
            recvcounts_for_n[i] = real_n * m;
            displs_for_n[i] = m * (i * real_n + buf_size) + m;
            }
        }

        real_n = recvcounts_for_n[ProcRank] / m;
        from_n = (displs_for_n[ProcRank] - m) / m ;

        //делим между процессами построчно, вычитаем строки
        for (int j=i+1+from_n;j<i+1+from_n+real_n;j++)
        {
            if (major_elements_array[j-i]==major_elements_array[0])
            {
                buf = (*(matrix+j*m + major_elements_array[j-i]-1)) / (*(matrix+i*m + major_elements_array[0]-1));


                    for (int k=major_elements_array[0]-1;k<m;k++)
                    {
                        (*(matrix+j*m + k)) -= (*(matrix+i*m + k)) * buf;
                    }

            }
        }

        free(major_elements_array);

        memcpy(buf_send, buf_matrix + m + from_n*m, real_n * m * sizeof(double));
        MPI_Allgatherv(buf_send, real_n * m, MPI_DOUBLE, buf_matrix, recvcounts_for_n, displs_for_n, MPI_DOUBLE, MPI_COMM_WORLD);

        buf_matrix = matrix+(i+1)*m;
        major_elements_array = find_major_elements_array(buf_matrix,n-i-1,m);

        // поиск нулевых строк и обновление массива главных элементов
        n -= zero_string_to_end(buf_matrix,n-i-1,m,major_elements_array);
    }

    free(buf_send);
    free(major_elements_array);
    return n;
}


int find_rank(double * matrix, int n, int m)
{
    return to_trapec_matrix(matrix,n,m);
}


int main(int argc, char **argv)
{
    assert(argc==2);
    FILE * myfile = NULL;
    FILE * output_file = fopen("output_file.txt","w+");
    n = 0,m = 0; // sizes of matrix
    double * matrix = NULL;

    double start_time, end_time;

    int res1 = 0;

    myfile = fopen(argv[1],"r");

    fscanf(myfile,"%d%d",&n,&m);

    matrix = get_matrix(n,m,myfile);

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    start_time = MPI_Wtime();

    real_m = m / ProcNum;
    int buf_size = m % ProcNum;

    //сколько столбцов
    recvcounts_for_m = calloc(ProcNum, sizeof(int));

    //начиная с какого
    displs_for_m = calloc(ProcNum, sizeof(int));

    for (int i=0; i<ProcNum; i++)
    {

        if (i < buf_size)
        {
            recvcounts_for_m[i] = real_m + 1;
            displs_for_m[i] = (i * real_m + i);
        }
        else
        {
            recvcounts_for_m[i] = real_m;
            displs_for_m[i] = (i * real_m + buf_size);
        }

    }

    from_m = displs_for_m[ProcRank];
    real_m = recvcounts_for_m[ProcRank];

    res1 = find_rank(matrix,n,m);

    end_time = MPI_Wtime();

    if (ProcRank == 0)
    {
        printf("Result = %d\n",res1);
        output_matrix(matrix,n,m,output_file);
    }


    fclose(myfile);
    fclose(output_file);
    free(matrix);
    MPI_Finalize();

    return 0;
}
