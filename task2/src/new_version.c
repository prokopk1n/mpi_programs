#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <mpi.h>
#include <string.h>
#include <signal.h>
#include <time.h>

#define N 10
#define accuracy 0.00001

#define create_checkpoint(func_call, matrix, size) \
    {double * buf_checkpoint = calloc(size, 1);\
    memcpy(buf_checkpoint , matrix, size);\
    do{memcpy(matrix, buf_checkpoint , size);error = 0; func_call;}while(error == 1);\
    free(buf_checkpoint);}

#define __error_situation__ if (ProcRank == N-1 && rand()%10 == 0) \
    {printf("Myrank = %d Line=%d and I exit\n", ProcRank, __LINE__); raise(SIGKILL);}

//у каждого процесса актуальная матрица
//m - столбцы, n - строки


int ProcNum = 0;
int ProcRank  = 0;
int real_m = 0,  from_m = 0, n, m, error = 0;
int * recvcounts_for_m = NULL, * displs_for_m = NULL;
MPI_Comm comm_world = MPI_COMM_WORLD;


//real_m - amount elements to be handled, from_m - from which elem
void calculate_parametres_for_allgather()
{
    real_m = m / ProcNum;
    int buf_size = m % ProcNum;

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
}


void verbose_errhandler(MPI_Comm *comm, int *perr, ...) 
{
    int rank, size, len, nf, eclass;
    int err = *perr; 
    MPI_Group group_c, group_f; 
    int *ranks_gc, *ranks_gf;
    error = 1; 

    printf("My rank %d and I in verbose_errhandler\n", ProcRank);
    MPI_Error_class(err, &eclass); 
    if( MPIX_ERR_PROC_FAILED != eclass && eclass != MPIX_ERR_REVOKED) { 
        MPI_Abort(*comm, err); 
    }
    if (ProcRank == 0)
        printf("ERROR OCCURED!\n");

    MPIX_Comm_shrink(*comm, &comm_world);

    MPI_Comm_rank(comm_world, &ProcRank);
    MPI_Comm_size(comm_world, &ProcNum);

    calculate_parametres_for_allgather();
}

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
void find_major_elements_array(double * matrix,int n, int m, int ** major_array)
{
    if (*major_array != NULL)
        free(*major_array);
    
    int * array = calloc(n,sizeof(int));
    int * buf_send = calloc(n,sizeof(int));

    __error_situation__

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

    __error_situation__

    int from_n = displs[ProcRank];
    real_n = recvcounts[ProcRank];

    //printf("real_n = %d, from_n = %d\n", real_n, from_n);
    for (int i=from_n;i<from_n + real_n;i++)
    {
        array[i] = find_major_element(matrix + i*m, m);
    }

    __error_situation__

    memcpy(buf_send, array+from_n, real_n * sizeof(int));
    if (!error)
        MPI_Allgatherv(buf_send,real_n,MPI_INT,array,recvcounts,displs, MPI_INT, comm_world);
    free(buf_send);

    *major_array = array;
}


//меняет нулевую и i-ую строки местами, m - длина строки
void swap_strings(double * matrix,int i,int m)
{
    double buf;
    double * buffer = calloc(real_m, sizeof(double));
    double * buf_send = calloc(m, sizeof(double));

    //меняем нужные нам части
    memcpy(buffer, matrix + i*m + from_m, real_m * sizeof(double));
    memcpy(matrix + i*m + from_m, matrix + from_m, real_m * sizeof(double));
    memcpy(matrix + from_m, buffer, real_m * sizeof(double));

    __error_situation__

    memcpy(buf_send, matrix + from_m, real_m * sizeof(double));
    if (!error)
        MPI_Allgatherv(buf_send, real_m, MPI_DOUBLE, matrix, recvcounts_for_m, displs_for_m, MPI_DOUBLE, comm_world);

    if (!error)
    {
        __error_situation__
        memcpy(buf_send, matrix + i*m + from_m, real_m * sizeof(double));
        MPI_Allgatherv(buf_send, real_m, MPI_DOUBLE, matrix + i*m, recvcounts_for_m, displs_for_m, MPI_DOUBLE, comm_world);
    }

    free(buf_send);
    free(buffer);
}

//подготавливаем ведуший элемент (строка, у которой самый левый ненулевой элемент выносится наверх матрицы)
void step_one(double * matrix, int n, int m, int * array)
{
    int min = array[0];

    for (int i=1;i<n;i++)
    {
        if (array[i] && min>array[i])
        {
            create_checkpoint(swap_strings(matrix,i,m), matrix, m*n*sizeof(double));
            min = array[i];
            array[i] = array[0];
            array[0] = min;
        }
    }
}


//меняет строки н1 и н2 местами, где н1 - нулевая строка
void swap_string_with_zero(double * matrix, int n1, int n2, int m)
{
    if (n1==n2)
        return;

    __error_situation__

    double * buf_send = calloc(real_m, sizeof(double));

    memcpy(matrix+n1*m + from_m, matrix+n2*m + from_m, real_m * sizeof(double));
    memset(matrix+n2*m + from_m, 0, real_m * sizeof(double));

    memcpy(buf_send, matrix + n1*m + from_m, real_m * sizeof(double));
    if (!error)
        MPI_Allgatherv(buf_send, real_m, MPI_DOUBLE, matrix + n1*m, recvcounts_for_m, displs_for_m, MPI_DOUBLE, comm_world);
    if (!error)
    {
        memcpy(buf_send, matrix + n2*m + from_m, real_m * sizeof(double));
        MPI_Allgatherv(buf_send, real_m, MPI_DOUBLE, matrix + n2*m, recvcounts_for_m, displs_for_m, MPI_DOUBLE, comm_world);
    }

    free(buf_send);
}


//возвращает количество перемещенных строк
int zero_string_to_end(double * matrix,int n, int m, int * array)
{
    int amount = 0;
    int i = 0;
    while (i<=n-amount-1)
    {
        if (array[i]==0)
        {
            __error_situation__

            create_checkpoint(swap_string_with_zero(matrix,i,n-amount-1,m), matrix, n*m*sizeof(double));
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


void matrix_print(double * matrix, int n, int m)
{
    for (int i=0;i<n;i++)
    {
        for (int j=0;j<m;j++)
            printf("%lf ",*(matrix+i*m+j));
        printf("\n");
    }
    printf("\n");
}

//вычитание строк
void sub_strings(double* matrix, int* major_elements_array, int n, int m)
{
    double * buf_send = calloc(n * m,sizeof(double));
    int recvcounts_for_n[ProcNum];
    int displs_for_n[ProcNum];
    int from_n;
    int real_n = n / ProcNum;
    int buf_size = n % ProcNum;
    for (int i=0; i<ProcNum; i++)
    {
        if (i < buf_size)
        {
            recvcounts_for_n[i] = (real_n + 1) * m;
            displs_for_n[i] = i * (real_n + 1) * m;
        }
        else
        {
            recvcounts_for_n[i] = real_n * m;
            displs_for_n[i] = m * (i * real_n + buf_size);
        }
    }

    __error_situation__

    real_n = recvcounts_for_n[ProcRank] / m;
    from_n = displs_for_n[ProcRank] / m ;
    double buf;

    for (int j=from_n;j<from_n+real_n;j++)
    {
        if (j == 0)
            continue;
        if (major_elements_array[j]==major_elements_array[0])
        {
            buf = (*(matrix+j*m + major_elements_array[j]-1)) / (*(matrix + major_elements_array[0]-1));


                for (int k=major_elements_array[0]-1;k<m;k++)
                {
                    (*(matrix+j*m + k)) -= (*(matrix + k)) * buf;
                }

        }
    }
    memcpy(buf_send, matrix + from_n*m, real_n * m * sizeof(double));
    if (!error)
        MPI_Allgatherv(buf_send, real_n * m, MPI_DOUBLE, matrix, recvcounts_for_n, displs_for_n, MPI_DOUBLE, comm_world);
}


int to_trapec_matrix(double * matrix, int n, int m)
{
    int * major_elements_array = NULL;

    create_checkpoint(find_major_elements_array(matrix,n,m, &major_elements_array), matrix, m*n*sizeof(double));
    n -= zero_string_to_end(matrix,n,m,major_elements_array);

    double * buf_matrix = matrix;
    double * buf_send = calloc(n * m,sizeof(double));

    int i = 0;
    while (i<n-1)
    {

        __error_situation__

        step_one(buf_matrix, n-i, m, major_elements_array);
        create_checkpoint(sub_strings(buf_matrix, major_elements_array, n-i, m), buf_matrix, (n-i)*m*sizeof(double));
        free(major_elements_array);
        major_elements_array = NULL;
        i+=1;

        buf_matrix = matrix+i*m;
        create_checkpoint(find_major_elements_array(buf_matrix,n - i,m, &major_elements_array), buf_matrix, m*n*sizeof(double));
        n -= zero_string_to_end(buf_matrix,n - i,m,major_elements_array);
    }

    free(buf_send);
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
    MPI_Errhandler errh;
    double start_time, end_time;

    int res1 = 0;

    myfile = fopen(argv[1],"r");
    fscanf(myfile,"%d%d",&n,&m);
    matrix = get_matrix(n,m,myfile);
    fclose(myfile);

    MPI_Init(NULL, NULL);
    MPI_Comm_size(comm_world, &ProcNum);
    MPI_Comm_rank(comm_world, &ProcRank);

    srand(time(NULL) + ProcRank); 

    MPI_Comm_create_errhandler(verbose_errhandler, &errh);
    int res = MPI_Comm_set_errhandler(comm_world, errh);
    if (res != MPI_SUCCESS)
        printf("HERE!!!\n");

    start_time = MPI_Wtime();

    recvcounts_for_m = calloc(ProcNum, sizeof(int));
    displs_for_m = calloc(ProcNum, sizeof(int));

    calculate_parametres_for_allgather();

    res1 = find_rank(matrix,n,m);

    end_time = MPI_Wtime();

    if (ProcRank == 0)
    {
        printf("Result = %d Time = %f\n",res1, end_time - start_time);
        output_matrix(matrix,n,m,output_file);
    }

    fclose(output_file);
    free(matrix);
    MPI_Finalize();

    return 0;
}
