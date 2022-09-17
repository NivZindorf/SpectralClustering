#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "spkmeans.h"
#define INVALID "Invalid Input!\n"
#define ERROR "An Error Has Occurred\n"

typedef struct {
double ** centroid0;
double ** centroid1;
int * counters;
int prev; /* if prev=0 then centroid0 is the prev */
}Centroids;

void print_matrix(double ** matrix,int n, int m);
double ** create_matrix(int n,int m);
double ** extract_data( char filename[],int * dim_p,int *vec_counter_p);
static void find_centroids(Centroids * centroids,double ** data,int k, int dimensional,int vec_counter);
static Centroids* create_centroids(double ** cent, int k, int dimensional);
int fit(double** data, double** cent,int k,int dim,int vec_count);
static void compute (Centroids * centroids,double ** data, int k, int dimensional,int max_iter,double EPS,int vec_counter);
static double distance(double * x,double * y, int dimensional);
static int deltamiu(Centroids * centroids,int k , int dimensional,double EPS);
void to_weighted(double ** weight, double ** vectors,int dim,int vec_counter);
void to_diagonal(double ** diagoanl,double ** weight,int vec_counter);
void matrixsqrt(double ** diagonal,int vec_counter);
void to_lnorm(double ** lnorm,double ** diagoanl,double ** weight,int vec_counter);
void matrix_mult(double ** A, double ** B, double ** tmp, int vec_counter);
double off(double ** A,int vec_counter);
void find_max(double ** mat,int *i_p,int* j_p,int vec_counter);
int to_jacobian(double ** jacob,double ** lnorm,int vec_counter);
void transpose(double** matrix, double ** trans, int n, int m);
int compare(const void *p1,const void *p2);
int eigengap(double ** jacobi,double ** T,int vec_counter,int k);
void free_matrix(double ** vectors,double ** weight,double ** diagonal,double ** lnorm,double ** jacobi );

int main(int argc, char *argv[]){
    char * options[5]={"spk","wam","ddg","lnorm","jacobi"};
    int i;
    int chosen;
    int res;
    char * goal;
    char * filename;
    double ** vectors ;
    double ** weight;
    double ** diagonal;
    double ** lnorm;
    double ** jacob;
    int dim,vec_counter;
    if (argc ==3){
        goal = argv[1];
        filename= argv[2];
    }
    else{
        printf(INVALID);
        return 1;
    }
    chosen =0;
    for(i=0; i<5; i++){
        if(strcmp(goal,options[i])==0){
            chosen+=1;
        }
    }
    if (chosen == 0){
        printf(INVALID);
        return 1;
    }
    

    vectors = extract_data(filename,&dim,&vec_counter);
    if (vectors == NULL){
        return 1;
    }
    if(strcmp(goal,"jacobi")==0){
        jacob = create_matrix(vec_counter+1, vec_counter);
        if(jacob ==NULL){
            free_matrix(vectors,NULL,NULL,NULL,NULL);
            printf(ERROR);
            return 1;
        }
        res= to_jacobian(jacob,vectors, vec_counter);
        if (res == 1){
            free_matrix(vectors,NULL,NULL,NULL,jacob);
            printf(ERROR);
            return 1;
        }
        print_matrix(jacob,vec_counter+1,vec_counter);
        free_matrix(vectors,NULL,NULL,NULL,jacob);
        return 0;
    }
    
    weight = create_matrix(vec_counter,vec_counter);
    if(weight == NULL) {
        free(vectors[0]);
        free(vectors);
        printf(ERROR);
        return 1;
    }
    to_weighted(weight,vectors, dim, vec_counter);
    if(strcmp(goal,"wam")==0){
        print_matrix(weight,vec_counter,vec_counter);
        free_matrix(vectors,weight,NULL,NULL,NULL);
        return 0;
    }
    diagonal = create_matrix(vec_counter,vec_counter);
    if(diagonal ==NULL){
        free_matrix(vectors,weight,NULL,NULL,NULL);
        printf(ERROR);
        return 1;
    }
    to_diagonal(diagonal, weight, vec_counter);
    if(strcmp(goal,"ddg")==0){
        print_matrix(diagonal,vec_counter,vec_counter);
        free_matrix(vectors,weight,diagonal,NULL,NULL);
        return 0;
    }
    lnorm = create_matrix(vec_counter,vec_counter);
    if(lnorm ==NULL){
        free_matrix(vectors,weight,diagonal,NULL,NULL);
        printf(ERROR);
        return 1;
    }
    to_lnorm(lnorm,diagonal,weight,vec_counter);
    if(strcmp(goal,"lnorm")==0){
        print_matrix(lnorm,vec_counter,vec_counter);
        free_matrix(vectors,weight,diagonal,lnorm,NULL);
        return 0;
    }
    
    return 0;
}

/*function which free the memory of all the given matrices*/
void free_matrix(double ** vectors,double ** weight,double ** diagonal,double ** lnorm,double ** jacobi ){
    if (vectors != NULL){
    free(vectors[0]);
    free(vectors);
    } 
    if (weight != NULL) {
        free(weight[0]);
        free(weight);
    }
    if (diagonal != NULL) {
        free(diagonal[0]);
        free(diagonal);
    }
    if (lnorm != NULL) {
        free(lnorm[0]);
        free(lnorm);
    }
    if (jacobi != NULL) {
        free(jacobi[0]);
        free(jacobi);
    }
}


void print_matrix(double ** matrix,int n, int m){
    int i,j;
    for(i=0;i<n;i++){
        for(j=0;j<m-1;j++){
            printf("%.4f,",matrix[i][j]);
        }
        printf("%.4f\n",matrix[i][m-1]);
    }
}


double ** create_matrix(int n,int m){
    int i;
    double *p;
    double **vectors;
    p = calloc(n*m, sizeof(double)); /*creating the data matrix */
    vectors = calloc(n,sizeof(double *));
    if (p==NULL || vectors == NULL){
        if(p!= NULL) free(p);
        if(vectors != NULL) free(vectors);
        return NULL;
    }
    for(i=0 ; i<n ; i++){
        vectors[i] = p+i*m;
    }
    return vectors;
}
double ** extract_data( char filename[],int * dim_p,int *vec_counter_p){ /* if the function return NULL then error occurred  */
    int i;
    int j;
    int c;
    int dim = 1;
    int vec_counter = 1;
    double *p;
    double **vectors;
    FILE *ifp=NULL;
    ifp = fopen(filename,"r");
    if(ifp == NULL){
        printf(ERROR);
        return NULL;
    }
    while((c=fgetc(ifp))!= '\n') /* check the dimensional of the vector */
    {
        if(c== ',')
        {
            dim++;
        }
    }
    while((c=fgetc(ifp))!= EOF){ /* check how many vectors */
        if(c =='\n')
        {
            vec_counter++;
        }
    }

    p = calloc(dim*vec_counter, sizeof(double)); /*creating the data matrix */
    vectors = calloc(vec_counter,sizeof(double *));
    if (p==NULL || vectors == NULL){
        if(p != NULL) free(p);
        if(vectors != NULL) free(vectors);
        fclose(ifp);
        printf (ERROR);
        return NULL;
    }
    for(i=0 ; i<vec_counter ; i++){
    vectors[i] = p+i*dim;
    }
    fseek(ifp,0,SEEK_SET);
    for (i=0;i<vec_counter;i++){
        for (j=0 ; j<(dim-1);j++){
            fscanf(ifp,"%lf,",&vectors[i][j]);
        }
        fscanf(ifp,"%lf",&vectors[i][dim-1]); 
    }
    fclose(ifp);
    *dim_p=dim;
    *vec_counter_p = vec_counter;
    return vectors;
}


/*----------------------------kmeans algoritm---------------------------------*/
int fit(double** data, double** cent,int k,int dim,int vec_count){
    int i,j;
    double **centroid1;
    int * counters;
    Centroids* centroids;

    centroids = create_centroids(cent, k, dim);
    if (centroids == NULL){
        free(cent); 
        free(data);
        return 1;
    }
    compute(centroids,data,k,dim,300,0,vec_count);
    if (centroids->prev == 1){
        for (i=0;i<k;i++){
            for(j=0;j<dim;j++){
                centroids->centroid0[i][j] = centroids->centroid1[i][j];
            }
        }
    }
    centroid1 = centroids->centroid1;
    counters = centroids->counters;
    free(cent);
    free(data);
    free(centroid1[0]);
    free(centroid1);
    free(counters);
    free(centroids);
    return 0;
}

static Centroids* create_centroids(double ** cent, int k, int dimensional){
    int i;
    double *p1 = calloc(dimensional*k, sizeof(double)); 
    double **centroid1 = calloc(k,sizeof(double *));
    int * counters = calloc(k,sizeof(int));
    Centroids* ret =malloc(sizeof(Centroids));
    if ( p1 == NULL || centroid1 == NULL || counters == NULL || ret == NULL){
        if (p1 != NULL) free(p1);
        if(centroid1 != NULL) free(centroid1);
        if(counters != NULL) free(counters);
        if (ret != NULL) free(ret);
        return NULL;
    }
    for(i=0 ; i<k ; i++){
        centroid1[i] = p1+i*dimensional;
    }
    
    ret->centroid0 = cent;
    ret->centroid1 = centroid1;
    ret->counters = counters;
    ret->prev = 0;
    return ret;
}

/*computing the centroids by doing max_iter iterations of matching them or until the centroids don't change*/
static void compute(Centroids* centroids,double ** data, int k, int dimensional,int max_iter,double EPS,int vec_counter){
    int i;
    int j;
    int count=0;
    double **new;
    
    do{
        count++;
        find_centroids(centroids,data,k,dimensional,vec_counter);
        new = centroids->prev ==0 ? centroids->centroid1: centroids->centroid0;
        for (i=0;i<k;i++){
            for (j=0;j<dimensional;j++){
                if (centroids->counters[i] != 0){
                    new [i][j] /= (centroids->counters[i]);
                }
            }
        }
    }while(deltamiu(centroids,k,dimensional,EPS) && count< max_iter); 
}

static void find_centroids(Centroids * centroids,double ** data,int k, int dimensional,int vec_counter){
     int i;
     int j;
     int index;
     double minsum;
     double cursum;
     double ** prev;
     double ** new;
     switch(centroids->prev){
         case 0:
            prev = centroids->centroid0;
            new = centroids->centroid1;
            break;
        case 1: 
            prev = centroids->centroid1;
            new = centroids->centroid0;
            break;
     }
     for(i=0;i<vec_counter;i++){
         minsum=distance(data[i],prev[0],dimensional);
         index=0;
         for(j=1;j<k;j++){
             cursum=distance(data[i],prev[j],dimensional);
             if (cursum<minsum){
                 minsum=cursum;
                 index=j;
             }
         }
         centroids->counters[index]++;
         for (j=0;j<dimensional;j++){
             new[index][j] += data[i][j];
         }
     }
}
/* actually compute distance^2 between vectors*/
static double distance(double * x,double * y, int dimensional){ 
    int i;
    double dist=0.0;
    for (i=0;i<dimensional;i++){
        dist += pow(x[i]-y[i],2);
    }
    return dist;
}

/*checking if need more itteration*/
static int deltamiu(Centroids * centroids,int k , int dimensional,double EPS){
    int ret=0;
    int i;
    int j;
    double ** prev;
    for (i=0;i<k;i++){
        if (sqrt(distance(centroids->centroid0[i], centroids->centroid1[i],dimensional))> EPS){
            ret=1;
            break;
        }
    }
    prev = centroids->prev ==0 ? centroids->centroid0: centroids->centroid1;
    for (i=0;i<k;i++){
        centroids->counters[i]=0;
        for(j=0;j<dimensional;j++){
            prev[i][j]=0;
        }
    }
    centroids->prev = !(centroids->prev);
    return ret;
}

/* ----------------The weighted Adjacency Matrix--------------------------*/
void to_weighted(double ** weight, double ** vectors,int dim,int vec_counter){
    int i,j;
    double tmp;
    for (i=0;i<vec_counter;i++){
        for (j=0;j<i;j++){
            tmp = sqrt(distance(vectors[i],vectors[j],dim));
            tmp = (-(tmp/2));
            tmp = exp(tmp);
            weight[i][j]= tmp;
            weight[j][i]= tmp;
        }
    }
}
/* ----------------The Diagonal Degree Matrix--------------------------*/
void to_diagonal(double ** diagoanl,double ** weight,int vec_counter){
    int i,j;
    double sum;
    for (i=0;i<vec_counter;i++){
        sum=0;
        for (j=0;j<vec_counter;j++){
            sum += weight[i][j];
        }
        diagoanl[i][i]= sum;
    }
}
void matrixsqrt(double ** diagonal,int vec_counter){
    int i;
    for(i=0;i<vec_counter;i++){
        diagonal[i][i]= 1/sqrt(diagonal[i][i]);
    }
}

/* ----------------The lnorm Matrix --------------------------*/
void to_lnorm(double ** lnorm,double ** diagoanl,double ** weight,int vec_counter){
    int i,j;
    matrixsqrt(diagoanl, vec_counter);
    for(i=0;i<vec_counter;i++){
        for(j=0;j<vec_counter;j++){
            lnorm[i][j]= diagoanl[i][i]*weight[i][j];
        }
    }
    for(i=0;i<vec_counter;i++){
        for(j=0;j<vec_counter;j++){
            lnorm[i][j]= (i==j)?  (1-(diagoanl[j][j]*lnorm[i][j])) : -(diagoanl[j][j]*lnorm[i][j]);
        }
    }

}
/* -------------------- Jacobian algo ---------------------- */
int to_jacobian(double ** jacob,double ** lnorm,int vec_counter){ 
    int i,j,sign,x,y;
    int count=0;
    double teta,t,c,s,offbefore,offafter,temp;
    double ** p;
    double ** v;

    p = create_matrix(vec_counter,vec_counter);
    v = create_matrix(vec_counter,vec_counter);
    if( p == NULL || v ==NULL){
        if(p!=NULL) {
            free(p[0]);
            free(p);}
        if(v!=NULL) {
            free(v[0]);
            free(v);}
        return 1;
    }
    for (x = 0; x < vec_counter; x++)
    {
        p[x][x] = 1;
        v[x][x] = 1;
    }
    offafter=off(lnorm,vec_counter);
    do{
        count++;
        find_max(lnorm,&i,&j,vec_counter);
        if(i==-1) break;
        teta = (lnorm[j][j]-lnorm[i][i])/(2*lnorm[i][j]);
        sign = (teta>=0)? 1:-1;
        t = sign/(fabs(teta)+sqrt(pow(teta,2)+1));
        c = 1/(sqrt(pow(t,2)+1));
        s = t*c;
        p[i][i] = c;
        p[j][j] = c;
        p[i][j] = s;
        p[j][i] = -s;
        offbefore = offafter;

        for(x=0; x<vec_counter;x++){
            if(x!=i && x!=j){
                temp = lnorm[x][i];
                lnorm[x][i] = (c*lnorm[x][i])-(s*lnorm[x][j]);
                lnorm[i][x] = lnorm[x][i];
                lnorm[x][j] = (c*lnorm[x][j])+(s*temp);
                lnorm[j][x] = lnorm[x][j];
            }
        }
        temp = lnorm[i][i];
        lnorm[i][i]= (pow(c,2)*lnorm[i][i])+(pow(s,2)*lnorm[j][j])-(2*s*c*lnorm[i][j]);
        lnorm[j][j]=(pow(s,2)*temp)+(pow(c,2)*lnorm[j][j])+(2*s*c*lnorm[i][j]);
        lnorm[i][j] = 0;
        lnorm[j][i] = 0;
        offafter = off(lnorm,vec_counter);

        for (x = 0; x < vec_counter; x++){
            temp = v[x][i];
            v[x][i] = (c*v[x][i])-(s*v[x][j]);
            v[x][j] = (s*temp)+(c*v[x][j]);
        }
        
        p[i][i] = 1;
        p[j][j] = 1;
        p[i][j] = 0;
        p[j][i]= 0;
    }
    while(!(offbefore-offafter<= 0.00001 || count ==100));
    for(x=0;x<vec_counter;x++){
        jacob[0][x]= lnorm[x][x];
    }
    for(x=0;x<vec_counter;x++){
        for(y=0;y<vec_counter;y++){
            jacob[x+1][y] = v[x][y]; 
        }
    }
    free(p[0]);
    free(p);
    free(v[0]);
    free(v);
    return 0;
}

/*calculate the offset of the matrix given for jacobi*/
double off(double ** A,int vec_counter){
    int i,j;
    double sum=0;
    for(i=1;i<vec_counter;i++){
        for(j=0;j<i;j++){
            sum += 2*pow(A[i][j],2);
        }
    }
    return sum;
}
/* find the Ai,j with max abs value, return -1 if diagonal*/
void find_max(double ** mat,int *i_p,int* j_p,int vec_counter){
    double max=0;
    int i = -1;
    int j = -1;
    int x,y;
    for (x=0;x<vec_counter;x++){
        for(y=x+1;y<vec_counter;y++){
            if(fabs(mat[x][y])>max){
                i=x;
                j=y;
                max = fabs(mat[x][y]);
            }
        }
    }
    *i_p =i;
    *j_p=j;
}

/*-------------The eigengap Heurisitic---------------------*/
/* a comparator for qsort*/
int compare(const void *p1,const void *p2){
    const double *x = *(double **)p1;
    const double *y = *(double **)p2;
    if (x[1] == y[1]) {
        return (int)(x[0]-y[0]);}
    if(x[1]>y[1]) return -1; 
    return 1; 
}

int eigengap(double ** jacobi,double ** T,int vec_counter,int k){
    double ** trans;
    double *p;
    int i,j;
    double max=0, sum;
    trans = create_matrix(vec_counter,vec_counter+2);
    p = trans[0];
    if (trans ==NULL){
        return -1;
    }
    /*transposing the matrix in order to get in-place sorting */
    transpose(jacobi,trans,vec_counter+1,vec_counter);
    for(j=0;j<vec_counter;j++){
        trans[j][0]=j;
    }  
    qsort(trans,vec_counter,sizeof(trans[0]),compare);
    if (k==0){
        for (j=0;j<=(vec_counter/2)-1;j++){
            if (fabs(trans[j][1] - trans[j+1][1])>max){
                max = fabs(trans[j][1] - trans[j+1][1]);
                k=j+1;
            }
        }
    }
    for (i=0;i<vec_counter;i++){
        sum=0;
        for(j=0;j<k;j++){
            T[i][j] = trans[j][i+2];
            sum += pow(T[i][j],2);
        }
        if(sum != 0){
            sum = sqrt(sum);
            for(j=0;j<k;j++){
                T[i][j] /= sum;
            }
        }
        
    }

    free(p);
    free(trans);
    return k;
}

void transpose(double** matrix, double ** trans, int n,int m){
    int i,j;
    for(i=1;i<n+1;i++){
        for(j=0; j<m;j++){
            trans[j][i]=matrix[i-1][j];
        }
    }
}

