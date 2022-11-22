#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>

using namespace std;

const int NNinputs = 12;
const int NNhiddens = 3;

const int windowSize = 4;
const int span = 2;


class NN {
    private:
        float weights_input[NNinputs+1][NNhiddens];     // 9 input + 3 imput + 1 bias
        float weights_hidden[NNhiddens+1];              // 3 hidden + 1 bias
    public:
        bool rulezNN(float* neigh);
        void loadWeights(float* weights);
        void printWeights();
};



#define MAX_DIM 200

bool campo[MAX_DIM][MAX_DIM][3];

int buffer = 6;

bool* getNeighbors(bool* neigh, int x, int y, int iteration);
bool rulez(float* neigh);
bool checkCampo(int dim, int iteration);
void readCampo(ifstream& inFile, int dim);
void printCampo(ofstream& outFile, int dim);
float* readWeights(ifstream& inWeights, float* weights);
int countParticles(int iteration, int dim, int posx, int posy);

int countVar(int n_particles, int tot_particles, int iteration);
int countDiff(int iteration, int dim);
float countAggregation(int iteration, int dim);


bool NN::rulezNN(float* neigh){
    float* hidden_decisions = new float[NNhiddens];
    float output_decision = 0.0;


    for (int i = 0; i < NNhiddens; i++){
        hidden_decisions[i] = 0.0;
        for (int j = 0; j < NNinputs; j++){
            hidden_decisions[i] += weights_input[j][i] * neigh[j];
        }
    }

    for (int i = 0; i < 3; i++){
        if (hidden_decisions[i] > weights_input[NNinputs][i]){
            hidden_decisions[i] = 1;
        }else{
            hidden_decisions[i] = 0;
        }
    }

    for (int i = 0; i < 3; i++){
        output_decision += hidden_decisions[i] * weights_hidden[i];
    }

    
    delete[] hidden_decisions;
    //delete[] neigh;
    
    if (output_decision > weights_hidden[NNhiddens]){
        return true;
    }else{
        return false;
    }
}

void NN::loadWeights(float* weights){
    int idx = 0;
    for (int i = 0; i < NNhiddens; i++){
        for (int j = 0; j < NNinputs+1; j++){
            weights_input[j][i] = weights[idx];
            idx++;
        }
    }
    for (int i = 0; i < NNhiddens+1; i++){
        weights_hidden[i] = weights[idx];
        idx++;
    }
}


void NN::printWeights(){
    cout << "----------------------------------" << endl;
    cout << "Input Weights:" << endl;
    for (int i = 0; i < NNhiddens; i++){
        for (int j = 0; j < NNinputs+1; j++){
            if (j == NNinputs){
                cout << "bias: ";
            }
            cout << weights_input[j][i] << " ";
        }
        cout << endl;
    }
    cout << "Hidden Weights:" << endl;
    for (int i = 0; i < NNhiddens+1; i++){
        if (i == NNhiddens){
            cout << "bias: ";
        }
        cout << weights_hidden[i] << " ";
    }
    cout << endl;
    cout << "----------------------------------" << endl;

}

bool* getNeighbors(bool* neigh, int x, int y, int iteration){
    int idx = 0;
    neigh = new bool[9];
    for (int i = x-1; i < x+2; i++){
        for (int j = y-1; j < y+2; j++){
            neigh[idx] = campo[i][j][iteration%buffer];
            idx++;
        }
    }
    return neigh;
}


bool rulez(bool* neigh){

    bool scelta = neigh[4];

    int somma = 0;
    for (int a = 0; a < 9; a++){
        if (a != 4){
            somma += neigh[a];
        }
    }
    if ((scelta) && (somma == 0)){
        return false;
    }

    if (somma > 0){
        if ((somma < 2) || (somma > 3)){
            scelta = false;
        }
        if (somma == 3){
            scelta = true;
        }
    }
    delete[] neigh;

    return scelta;
}

//Conta le particelle true nel campo
int countParticles(int iteration, int dim, int posx, int posy){

    int counter = 0;
    for (int i = 20+posx; i < 20+dim+posx; i++){
        for (int j = 20+posy; j < 20+dim+posy; j++){
            if (campo[i][j][iteration%buffer] == true){
                counter++;
            }
        }
    }
    return (counter);
}

//Calcola la varianza attuale nel campo
int countVar(int n_particles, int tot_particles, int iteration){
    int var = 0;


    var = pow((n_particles - (tot_particles/iteration)), 2);

    return var;
}

//Conta le particelle cambiate rispetto al campo precedente
int countDiff(int iteration, int dim){
    int diffs = 0;

    for (int i = 20; i < dim+20; i++){
        for (int j = 20; j < dim+20; j++){
            if (campo[i][j][iteration%buffer] ^ campo[i][j][iteration%buffer-1]){
                diffs++;
            }
        }
    }
    return (diffs);

}

float countAggregation(int iteration, int dim){
    float dev_std = 0;

    int Nelements = (dim-span)/span;
    int* agg;
    agg = new int[Nelements*Nelements];
    float mean = 0;
    int idx = 0;
    for (int a = 0; a < dim-span; a += span){
        for (int b = 0; b < dim-span; b += span){
            agg[idx] = countParticles(iteration, windowSize, a, b);
            mean += agg[idx];
            idx++;
        }
    }
    mean = mean/float(Nelements*Nelements);
    
    for (int i = 0; i < Nelements*Nelements; i++){
        dev_std += pow((float(agg[i]) - mean), 2);
        //cout << agg[i] << ", ";
    }
    //cout << endl;
    //cout << dev_std/(Nelements*Nelements) << endl;
    delete[] agg;
    return dev_std/float(Nelements*Nelements);
   
}


//Controlla se il campo attuale corrisponde con uno dei campi nel buffer
bool checkCampo(int dim, int iteration){    //true if there is one field repeating

    bool checks[buffer-1];
    bool out;

    for(int k = 0; k < buffer-1; k++){
        checks[k] = true;
    }

    for (int i = 20; i < dim+20; i++){
        for (int j = 20; j < dim+20; j++){
            for(int k = 1; k < buffer; k++){
                if(campo[i][j][iteration%buffer] ^ campo[i][j][(iteration-k)%buffer]){
                    checks[k-1] = false;
                }
            }
        }
    }

    for(int k = 0; k < buffer-1; k++){
        //cout << checks[k] << " ";
        out |= checks[k];
    }
    //cout << endl;

    // cout << "Decision: " << check1 << " or " << check2 << endl;
    return out;


}

float* readWeights(ifstream& inWeights, float* weights){
    weights = new float[43];
    int idx = 0;
    float x;
    //cout << "GENOTYPE:" << endl;
    while (inWeights >> x){
        //cout << idx << ": " << x << endl;
        weights[idx] = x;
        idx++;
    }
    //cout << endl;
    return weights;
}

void readCampo(ifstream& inFile, int dim){
    int i = 20;
    int j = 20;
    bool x;
    //cout << "INITIAL FIELD:" << endl;
    while (inFile >> x) {
        //cout << x << " ";
        campo[i][j][0] = x;
        j++;
        if (j == dim+20){
            j = 20;
            i++;
            //cout << endl;
        }
    }
}

void printCampo(ofstream& outFile, int dim){
    for (int i = 20; i < dim+20; i++){
        for (int j = 20; j < dim+20; j++){
            outFile << campo[i][j][0] << " ";
        }
        outFile << endl;
    }
}
