#include <iostream>
#include <fstream>
#include <cstdlib>
#include <unistd.h>


#include "gol.h"

using namespace std;

/*
- creare adattamento a NN (weights in input)
- lettura campo iniziale da file (deciso da python)
- lettura weights da file
- output max iterations (file o terminale)
*/



int main(int argc, char *argv[]){
    int dim, max_iterations;

    if (argc < 2){
        dim = 12;
    }else{
        dim = atoi(argv[1]);
    }

    if (argc < 3){
        max_iterations = 5000;
    }else{
        max_iterations = atoi(argv[2]);
    }

    cout << "DIM: " << dim << endl;
    cout << "MAX_ITERATIONS: " << max_iterations << endl;

    for (int c = 0; c < 3; c++){
        for (int i = 0; i < MAX_DIM; i++){
            for (int j = 0; j < MAX_DIM; j++){
                campo[i][j][c] = false;
            }
        }
    }
    for (int c = 0; c < 3; c++){
        for (int i = 20; i < dim+20; i++){
            for (int j = 20; j < dim+20; j++){
                campo[i][j][c] = false;
            }
        }
    }

    /*
    float weights[34] = {1, 1, 1, 1, 0, 1, 1, 1, 1, 3,
                          1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
                          -1, 1, 1, 1};
    */

    float* weights;
    bool check = false;

    srand (time(NULL));

    ifstream inWeights;

    inWeights.open("/../best.txt");


    if (!inWeights) {
        cerr << "Unable to open file input files, change directory in gol.cpp ";
        exit(1);   // call system to stop
    }

    weights = readWeights(inWeights, weights);

    NN net;

    net.loadWeights(weights);


    

    //printCampo(outFile, dim);
    int prev_iteration = 10000;

    for (int round = 0; round < 2; round ++){
    
    //random initialization
    for (int i = 20; i < dim+20; i++){
        for (int j = 20; j < dim+20; j++){
            if ((rand() % 10 + 1) < 3){
                campo[i][j][0] = true;
            }
        }
    }


    int iteration = 0;
    while(iteration < max_iterations){

        // cout << "----------- Iter: " << iteration << endl;
        
        for (int i = 20; i < dim+20; i++){
            for (int j = 20; j < dim+20; j++){
                if (campo[i][j][iteration%buffer]){
                    cout << "# ";
                }else{
                    cout << "  ";
                }
            }
            cout << "|" << endl;
        }
        

        bool* neigh;
        float* inputs;
        //copiaCampo(dim);
        for (int i = 20; i < dim+20; i++){
            for (int j = 20; j < dim+20; j++){
                // cout << "Point: (" << i << ", " << j << "):" << endl;
                neigh = getNeighbors(neigh, i, j, iteration);
                inputs = new float[NNinputs];
                for (int a = 0; a < NNinputs; a++){
                    if (a < 9){
                        inputs[a] = neigh[a];
                    }else{
                        inputs[a] = 0.0;
                    }
                }
                inputs[9] = countParticles(iteration, dim)/float(dim*dim);
                    

                if (net.rulezNN(inputs)){
                    campo[i][j][(iteration+1)%buffer] = true;
                }else{
                    campo[i][j][(iteration+1)%buffer] = false;
                }

                delete[] inputs;
            }
        }

        iteration++;
        usleep(120000);
        clearScreen();
        

        if (checkCampo(dim, iteration)){
            //cout << "TOT. ITERATIONS: " << iteration << endl;
            //outIterations << iteration << " " << n_particles/iteration;
            check = true;
        }
    }


    }

    

    return 0;
}


