#include<iostream>
#include<fstream>
#include<sstream>
#include<map>
#include<vector>

#include <flann/flann.hpp>

#include "Vec.h"

#include "VecDistance.h"

using namespace std;
using namespace flann;

//TODO: CHANGE WHEN NOT DEBUGING
const string VECTOR_DIR = "./data";
const int NUM_VEC_FILES = 1;
const char MAJOR_DELIM = '|';
const int VECTOR_SIZE = 500;

//loads files from VECTOR_DIR and updates pmid2vec
void loadVecs(map<string,Vec> & pmid2vec){
        
    for(int i = 0; i < NUM_VEC_FILES; i++){
        string filePath = VECTOR_DIR + "/vectors" + to_string(i);
        fstream fin(filePath,ios::in);
        string line;
        while(getline(fin,line)){
            if(line.size() > 0){
                try{
                    int barIndex = line.find(MAJOR_DELIM);
                    string pmid = line.substr(0, barIndex);
                    string vectorData = line.substr(barIndex + 1);
                    stringstream s;
                    s << vectorData;
                    vector<float> data;
                    float t;
                    while(s >> t) data.push_back(t);
                    Vec vec(data);

                    //if we need to merge
                    if(pmid2vec.find(pmid) != pmid2vec.end()){
                        pmid2vec[pmid] += vec;
                        pmid2vec[pmid] /= 2;
                    }else{//add new elem
                        pmid2vec[pmid] = vec;
                    }
                }
                catch(...){
                    //DO NOTHING
                }
            }
        }      
        fin.close();
    }
}

int main(int argc, char** argv)
{

    map<string,Vec> pmid2vec;
    
    
    loadVecs(pmid2vec);
    
    //row major order, map.size() rows and VECTOR_SIZE cols
    float * vecData = new float[pmid2vec.size()*VECTOR_SIZE];
    
    
    //cout<<"Fill Array" << endl;
    vector<string> pmids;
    int pmidCount = 0;
    for(auto val : pmid2vec){
        pmids.push_back(val.first);
        for(int i = 0 ; i < VECTOR_SIZE; i++){
            vecData[VECTOR_SIZE * pmidCount + i] = val.second.get(i);
        }
        pmidCount++;
    }
    
    //cout<<"Make FLANN Array" << endl;
    
    flann::Matrix<float> data(vecData,pmid2vec.size(),VECTOR_SIZE);
      
    //cout << "Make Index" << endl; 
    flann::Index<flann::L2<float> > index(data, flann::KDTreeIndexParams(16));
    index.buildIndex();
    
    vector<vector<int> > indicies;
    vector<vector<float> > dists;
    
    flann::SearchParams params(128);
    params.cores = 0; //automatic core selection
    index.knnSearch(data, indicies, dists, 10, params);
 
    for(int currIndex = 0; currIndex < indicies.size(); ++currIndex){
        string pmid = pmids[currIndex];
        for(int nIndex = 0; nIndex < indicies[currIndex].size(); ++nIndex){
            int neighborID = indicies[currIndex][nIndex];
            string nPMID = pmids[neighborID];
            
            float pairDist = dists[currIndex][nIndex];
            
            cout<<pmid << " " << nPMID << " " <<pairDist<< endl;
        }
    }
    
    delete[] vecData;
    
    return 0;
}