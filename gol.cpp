#include <iostream>
#include <fstream>
#include <cstdlib>



#include "gol.h"

using namespace std;


int main(int argc, char *argv[]){
    
    
    int dim = atoi(argv[1]);
    int max_iterations = atoi(argv[2]);
    // cout << "DIM: " << dim << endl;
    // cout << "MAX_ITERATIONS: " << max_iterations << endl;
    
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
    int n_particles = 0;
    int n_var = 0;
    float n_agg = 0.0;
    int cumulative_movement = 0;

    srand (time(NULL));

    ifstream inFile;
    ifstream inWeights;
    ofstream outFile;
    ofstream outIterations;
    
    inFile.open("initial_field.txt");
    inWeights.open("weights.txt");
    //outFile.open("initial_field.txt");
    outIterations.open("iterations.txt");

    if ((!inFile) || (!inWeights)) {
        cerr << "Unable to open file input files";
        exit(1);   // call system to stop
    }
    
    //readCampo(inFile, dim);
    weights = readWeights(inWeights, weights);
    
    NN net;

    net.loadWeights(weights);
    //net.printWeights();


    //printCampo(outFile, dim);
    int prev_iteration = 10000;
    int prev_particles = 0;
    int prev_var = 0;
    float prev_agg = 0.0;
    int outValue = 0;
    for (int round = 0; round < 2; round ++){           // 2 round per evitare outlayers
    //cout << "Round " << round << endl;

    n_particles = 0;
    n_var = 0;
    n_agg = 0.0;

        //random initialization of campo
        for (int i = 20; i < dim+20; i++){
            for (int j = 20; j < dim+20; j++){
                if ((rand() % 10 + 1) < 3){
                    campo[i][j][0] = true;
                }
            }
        }


        int iteration = 0;

        while(iteration < max_iterations){
            /*
            cout << "----------- Iter: " << iteration << endl;
            
            for (int i = 20; i < dim+20; i++){
                for (int j = 20; j < dim+20; j++){
                    cout << campo[i][j][iteration%3] << " ";
                }
                cout << endl;
            }
            */
            
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
                    inputs[9] = countParticles(iteration, dim, 0, 0)/float(dim*dim);
                    
                    
                    if (net.rulezNN(inputs)){
                        campo[i][j][(iteration+1)%buffer] = true;
                    }else{
                        campo[i][j][(iteration+1)%buffer] = false;
                    }
                    
                    delete[] inputs;
                }
            }

            iteration++;
            
            int par = countParticles(iteration, dim, 0, 0);
            n_particles += par;
            //n_var += countVar(par, n_particles, iteration);
            n_var += countDiff(iteration, dim);
            n_agg += countAggregation(iteration, dim);
            

            if (checkCampo(dim, iteration)){
                //cout << "TOT. ITERATIONS: " << iteration << endl;
                //outIterations << iteration << " " << n_particles/iteration;
                outValue = iteration;
                iteration = max_iterations;
                check = true;
            }
        }

        
        //n_diff = n_diff/144;
        //cout << n_diff << endl;

        if (check == false){
            //cout << "TOT. ITERATIONS: " << max_iterations << endl;
            //outIterations << max_iterations<< " " << n_particles/max_iterations;
            outValue = max_iterations;
        }

        if (outValue < prev_iteration){
            prev_iteration = outValue;
            prev_particles = n_particles;
            prev_var = n_var;
            prev_agg = n_agg;
        }
        

    }

    cout << "iterations: " << prev_iteration << endl;

    outIterations << prev_iteration << " " << prev_particles/(prev_iteration) << " " << prev_var/(prev_iteration) << " " << prev_agg/(prev_iteration);


    return 0;
}


