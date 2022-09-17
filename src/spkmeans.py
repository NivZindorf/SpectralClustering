import mykmeanssp
import sys
import numpy as np

def initial_centroids(data ,K):
    N= data.shape[0]
    np.random.seed(0)
    realindices = np.arange(data.shape[0])
    indices = np.zeros(shape = K)
    centroids = np.zeros(shape = (K,data.shape[1]))
    first = np.random.choice(realindices)
    centroids[0]=data[first,:]
    indices[0] = first
    for i in range(K-1):
        D_l = np.zeros(shape = N)
        D_sum = 0    
        for l in range(N):
            min1 = np.sum(np.power(data[l,:]-centroids[0,:],2))
            for j in range (1,i+1):
                curr = np.sum(np.power(data[l,:]-centroids[j,:],2))
                if curr<min1:
                    min1 = curr
            D_l[l]=min1
            D_sum += min1
        if D_sum != 0:
            P = D_l/D_sum
            rand = np.random.choice(realindices,p=P)
        else:
            rand =np.random.choice(realindices)
        centroids[i+1,:] = data[rand,:]
        indices[i+1] = rand
    return centroids ,indices

def print_matrix(matrix):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]-1):
            print("%.4f,"%matrix[i][j], end="")
        print("%.4f"%matrix[i][-1])

def check_output(matrix,res, to_print): #return True if need to exit
    if (res == 0):
        if(to_print):
            print_matrix(matrix)
            return True
        return False
    else:
        print("An Error Has Occured")
        return True
    
def main():
    try:
        if (len(sys.argv) == 4):
            MAX_ITER = 300
            K = int(float(sys.argv[1]))
            EPS = 0
            goal = sys.argv[2]
            INPUT = sys.argv[3]
            if (goal == 'spk' and (K != float(sys.argv[1]) or K<0) ):
                print("Invalid Input!")
                return
        else:
            print("Invalid Input!")
            return
        data = np.loadtxt(INPUT, delimiter=',')
        if(K<0 or K>=data.shape[0]):
            print("Invalid Input!")
            return
        # if the goal is jacobi, then we get a symmetic matrix as input
        if(goal=='jacobi'):
            jacob = np.ascontiguousarray(np.zeros(shape=(data.shape[0]+1,data.shape[0]), dtype=np.float64))
            res = mykmeanssp.to_jacobian(data,jacob,data.shape[0])
            check_output(jacob,res, True)
            return

        # if any other goal, then start running the algoritm and stop according to the specific goal
        elif(goal in ["wam","ddg","lnorm","spk"]):
            # computing the weighted matrix
            weighted = np.ascontiguousarray(np.zeros(shape=(data.shape[0],data.shape[0]), dtype=np.float64))
            res = mykmeanssp.to_weighted(data,weighted,data.shape[0],data.shape[1])
            to_print = (goal == "wam")
            if(check_output(weighted,res,to_print)):
                return
            
            # computing the diagonal matrix
            diagonal = np.ascontiguousarray(np.zeros(shape=(weighted.shape[0],weighted.shape[0]), dtype=np.float64))
            res = mykmeanssp.to_diagonal(weighted,diagonal,weighted.shape[0])
            to_print = (goal == "ddg")
            if(check_output(diagonal,res,to_print)):
                return
            
            # computing the lnorm matrix
            lnorm = np.ascontiguousarray(np.zeros(shape=(diagonal.shape[0],diagonal.shape[0]), dtype=np.float64))
            res = mykmeanssp.to_lnorm(weighted,diagonal,lnorm,weighted.shape[0])
            to_print = (goal == "lnorm")
            if(check_output(lnorm,res,to_print)):
                return
            
            # computing jacobi on the lnorm matrix
            jacob = np.ascontiguousarray(np.zeros(shape=(lnorm.shape[0]+1,lnorm.shape[0]), dtype=np.float64))
            res = mykmeanssp.to_jacobian(lnorm,jacob,lnorm.shape[0])
            if(res == 1):
                print("An Error Has Occured")
                return

            # computing the eigengap and finding K if needed
            T = np.ascontiguousarray(np.zeros(shape=(lnorm.shape[0],lnorm.shape[1]), dtype=np.float64))
            K = mykmeanssp.eigengap(jacob,T,jacob.shape[1],K)
            if(K == -1): # if the program failed it returns -1;
                print("An Error Has Occured")
                return
            T = T[:,:K]
            # running the kmeans++ algoritm on T
            cent , indices= initial_centroids(T,K)
            T = np.ascontiguousarray(T, dtype=np.float64)
            cent = np.ascontiguousarray(cent, dtype=np.float64)
            if(mykmeanssp.fit(MAX_ITER, EPS, T, cent) == 0):
                for i in range(len(indices)-1):
                    print(f"{int(indices[i])},", end="")
                print(f"{int(indices[-1])}")
                print_matrix(cent)
            else:
                print("An Error Has Occured")
                return
        else:
            print("Invalid Input!")
            return
    except:
        print("Invalid Input!")
        return

if __name__=="__main__":
    main()