# Deep Causal Reasoning for Recommender Systems

 The codes are associated with the following paper:
 >**Deep Causal Reasoning for Recommendations,**  
 >Yaochen Zhu, Jing Yi, Jiayi Xie and Zhenzhong Chen,  
 >ArXiv Preprints 2022. [[pdf]](https://arxiv.org/abs/2201.02088)

Note: To better understand Rubin and Pearl's causal framework discussed in this paper, check out our [new repo](https://github.com/yaochenzhu/awesome-books-for-causality) that summarizes relevant books of and disputes between the two most prominent schools of causal inference.

## Environment

 The codes are written in Python 3.6.5.  

- numpy == 1.16.3
- pandas == 0.21.0
- tensorflow-gpu == 1.15.0
- tensorflow-probability == 0.8.0

## Dataset Acquirement and Simulation

- **Acquire the movielens-1m and amazon-vg datasets:**  
    The original datasets can be found [[here]](https://grouplens.org/datasets/movielens/1m/) and [[here]](https://jmcauley.ucsd.edu/data/amazon/).  
 Preprocess the data with data_sim/raw/prepare_data.py.

- **Preprocess the original dataset:**
    cd to data_sim/raw folder, run   
    ```python prepare_data.py --dataset Name --simulate {exposure, ratings}```.

- **Fit the exposure and rating distribution via VAEs:**
    cd to data_sim folder, run   
    ```python train.py --dataset Name --simulate {exposure, ratings}```. 

- **Simulate the causal dataset under various confounding levels:**    
    ```python simulate.py --dataset Name --simulate {exposure, ratings}```. 

- **The simulated datasets are in casl/data folder**

## Fitting the Exposure and Rating Models
- **Split the simulated causal datasets into train/val/test:**  
    cd to casl_rec/data folder, run   
    ```python preprocess.py --dataset Name --split 5```.

- **Train the exposure model, conduct predictive check:**  
    ```python train_exposure.py --dataset Name --split [0-4]```

- **Infer the subsititute confounders:**   
    ```python infer_subs_conf.py --dataset Name --split [0-4]```

- **Train the potential rating prediction model:**   
    ```python train_ratings.py --dataset Name --split [0-4]```

- **Predict the scores for hold-out users:**   
    ```python evaluate_model.py --dataset Name --split [0-4]```

**For advanced argument usage, run the code with --help argument.**
