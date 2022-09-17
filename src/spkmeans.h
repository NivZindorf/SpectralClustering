/*typedef struct {
double ** centroid0;
double ** centroid1;
int * counters;
int prev; 
}Centroids;
void find_centroids(Centroids * centroids,double ** data,int k, int dimensional,int vec_counter);
Centroids* create_centroids(double ** data, double ** cent, int k, int dimensional);
void compute (Centroids * centroids,double ** data, int k, int dimensional,int max_iter,double EPS,int vec_counter);
double distance(double * x,double * y, int dimensional);
int deltamiu(Centroids * centroids,int k , int dimensional,double EPS);*/
int fit(double** data, double** cent,int k,int dim,int vec_count);
void to_weighted(double ** weight, double ** vectors,int dim,int vec_counter);
void to_diagonal(double ** diagoanl,double ** weight,int vec_counter);
void to_lnorm(double ** lnorm,double ** diagoanl,double ** weight,int vec_counter);
int to_jacobian(double ** jacob,double ** lnorm,int vec_counter);
int eigengap(double ** jacobi,double ** T,int vec_counter,int k);
