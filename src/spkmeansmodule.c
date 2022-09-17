#define PY_SSIZE_T_CLEAN 
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/ndarraytypes.h>
#include <stdio.h>
#include "spkmeans.h"

PyObject* fit_api(PyObject *self, PyObject *args);
PyObject* jacobian_api(PyObject *self, PyObject *args);
PyObject* weighted_api(PyObject *self, PyObject *args);
PyObject* diagonal_api(PyObject *self, PyObject *args);
PyObject* lnorm_api(PyObject *self, PyObject *args);
PyObject* eigen_api(PyObject *self, PyObject *args);

int fit_c(int max_iter,double EPS,PyArrayObject* maindata, PyArrayObject* centroidsdata);
double ** to_matrix(PyArrayObject * obj);

static PyMethodDef capiMethods [] = {
    {"fit",(PyCFunction) fit_api, METH_VARARGS,PyDoc_STR("C kmeans")},
    {"to_jacobian",(PyCFunction) jacobian_api,METH_VARARGS,PyDoc_STR("jacobian")},
    {"to_weighted",(PyCFunction) weighted_api,METH_VARARGS,PyDoc_STR("weighted matrix")},
    {"to_diagonal",(PyCFunction) diagonal_api,METH_VARARGS,PyDoc_STR("diagonal matrix")},
    {"to_lnorm",(PyCFunction) lnorm_api,METH_VARARGS,PyDoc_STR("lnorm matrix")},
    {"eigengap",(PyCFunction) eigen_api,METH_VARARGS,PyDoc_STR("eigengap and normilized matrix")},
    {NULL,NULL,0,NULL}
};
static struct PyModuleDef moduledef  = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp",
    "Python interface for the kmeans C library function",
    -1,
    capiMethods
};

PyMODINIT_FUNC PyInit_mykmeanssp(void){
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (!m){
        return NULL;
    }
    return m;
}

PyObject* jacobian_api(PyObject *self, PyObject *args){
    PyObject* maindata;
    PyObject* mainjacobi;
    double ** data;
    double ** jacobi;
    int result;
    int vec_counter;
    if (!PyArg_ParseTuple(args,"OOi",&maindata,&mainjacobi,&vec_counter)){
        return Py_BuildValue("i",1);
    }
    data = to_matrix((PyArrayObject *)maindata);
    jacobi = to_matrix((PyArrayObject *)mainjacobi);
    if (data == NULL || jacobi == NULL){
        if(data != NULL){
            free(data);
        }
        if(jacobi != NULL){
            free(jacobi);
        }
        return Py_BuildValue("i",1);
    }
    result = to_jacobian(jacobi,data,vec_counter);
    free(data);
    free(jacobi);
    return Py_BuildValue("i",result);
}

PyObject* weighted_api(PyObject *self, PyObject *args){
    PyObject* maindata;
    PyObject* mainweighted;
    double ** data;
    double ** weighted;
    int vec_counter;
    int dim;
    if (!PyArg_ParseTuple(args,"OOii",&maindata,&mainweighted,&vec_counter,&dim)){
        return Py_BuildValue("i",1);
    }
    data = to_matrix((PyArrayObject *)maindata);
    weighted = to_matrix((PyArrayObject *)mainweighted);
    if (data == NULL || weighted == NULL){
        if(data != NULL){
            free(data);
        }
        if(weighted != NULL){
            free(weighted);
        }
        return Py_BuildValue("i",1);
    }
    to_weighted(weighted,data,dim,vec_counter);
    free(data);
    free(weighted);
    return Py_BuildValue("i",0);
}

PyObject* diagonal_api(PyObject *self, PyObject *args){
    PyObject* maindata;
    PyObject* maindiagonal;
    double ** data;
    double ** diagonal;
    int vec_counter;
    if (!PyArg_ParseTuple(args,"OOi",&maindata,&maindiagonal,&vec_counter)){
        return Py_BuildValue("i",1);
    }
    data = to_matrix((PyArrayObject *)maindata);
    diagonal = to_matrix((PyArrayObject *)maindiagonal);
    if (data == NULL || diagonal == NULL){
        if(data != NULL){
            free(data);
        }
        if(diagonal != NULL){
            free(diagonal);
        }
        return Py_BuildValue("i",1);
    }
    to_diagonal(diagonal,data,vec_counter);
    free(data);
    free(diagonal);
    return Py_BuildValue("i",0);
}

PyObject* lnorm_api(PyObject *self, PyObject *args){
    PyObject* mainweighted;
    PyObject* maindiagonal;
    PyObject* mainlnorm;
    double ** weighted;
    double ** diagonal;
    double ** lnorm;
    int vec_counter;
    if (!PyArg_ParseTuple(args,"OOOi",&mainweighted,&maindiagonal,&mainlnorm,&vec_counter)){
        return Py_BuildValue("i",1);
    }
    weighted = to_matrix((PyArrayObject *)mainweighted);
    diagonal = to_matrix((PyArrayObject *)maindiagonal);
    lnorm = to_matrix((PyArrayObject *)mainlnorm);
     if (weighted == NULL || diagonal == NULL ||lnorm == NULL){
        if(weighted != NULL){
            free(weighted);
        }
        if(diagonal != NULL){
            free(diagonal);
        }
        if(lnorm != NULL){
            free(lnorm);
        }
        return Py_BuildValue("i",1);
    }
    to_lnorm(lnorm,diagonal,weighted,vec_counter);
    free(weighted);
    free(diagonal);
    free(lnorm);
    return Py_BuildValue("i",0);
}

PyObject* eigen_api(PyObject *self, PyObject *args){
    PyObject* mainjacobi;
    PyObject* mainT;
    double ** jacobi;
    double ** T;
    int vec_counter;
    int k;
    if (!PyArg_ParseTuple(args,"OOii",&mainjacobi,&mainT,&vec_counter,&k)){
        return Py_BuildValue("i",-1);
    }
    jacobi = to_matrix((PyArrayObject *)mainjacobi);
    T = to_matrix((PyArrayObject *)mainT);
    if(jacobi == NULL || T ==NULL){
        if(jacobi != NULL){
            free(jacobi);
        }
        if(T != NULL){
            free(T);
        }
        return Py_BuildValue("i",-1);
    }
    k = eigengap(jacobi,T,vec_counter,k);
    free(jacobi);
    free(T);
    return Py_BuildValue("i",k);
}

/* casting the Python Numpy Array to C array */
double ** to_matrix(PyArrayObject * obj){
    int i;
    double *pdata = (double*)PyArray_DATA(obj);
    int dimensional = PyArray_DIM(obj,1);
    int vec_counter = PyArray_DIM(obj,0);
    double **data = calloc(vec_counter, sizeof(double *));
    if (data == NULL){
        return NULL;
    }
    for(i=0; i<vec_counter ; i++){
        data[i] = pdata+i*dimensional;
    }
    return data;
}

/* the final kmeans++ part from HW 2 */
PyObject* fit_api(PyObject *self, PyObject *args){
    PyObject* maindata;
    PyObject* centroidsdata;
    double EPS;
    int max_iter;
    if (!PyArg_ParseTuple(args,"idOO",&max_iter,&EPS,&maindata,&centroidsdata)){
        return Py_BuildValue("i",1);
    }
    return Py_BuildValue("i",fit_c(max_iter,EPS,(PyArrayObject *)maindata ,(PyArrayObject *)centroidsdata ));

}

int fit_c(int max_iter,double EPS,PyArrayObject* maindata, PyArrayObject* centroidsdata){
    double *pdata = (double*)PyArray_DATA(maindata);
    int dimensional = PyArray_DIM(maindata,1);
    int vec_counter = PyArray_DIM(maindata,0);
    double *p_cent = (double*)PyArray_DATA(centroidsdata);
    int k = PyArray_DIM(centroidsdata,0);
    double **cent = calloc(k,sizeof(double *));
    double **data = calloc(vec_counter, sizeof(double *));
    if (cent == NULL || data ==NULL){
        if(data !=NULL) free(data);
        if (cent !=NULL) free (cent);
        return 1;
    }
    int i;
    for(i=0; i<vec_counter ; i++){
        data[i] = pdata+i*dimensional;
    }
    for(i=0; i<k ; i++){
        cent[i] = p_cent+i*dimensional;
    }
    fit(data, cent, k, dimensional, vec_counter);
    return 0;
}
