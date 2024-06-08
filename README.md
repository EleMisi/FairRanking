# FairRanking
This repository provides the code for reproducing the results obtained in the paper 
*Long-Term Fairness Strategies in Ranking with Continuous Sensitive Attributes* (currently under revision at AEQUITAS@ECAI24)

### Prerequisites

* Virtual environment with Python >=3.7
* Packages:
  ```
  pip install -r requirements.txt
  ```
  
### How to run
The script `run.py` replicates the results presented in the paper.
For Discrete Actions experiments run:

  ```
  python run.py --actions discrete
  ```

For Continuous Actions experiments run:

  ```
  python run.py --actions continuous
  ```

Results are stored in `results_group_weights` folder for Discrete Actions experiment and `results_polynomial_fn` for 
Continuous Actions experiment.\
Each folder contains a `records` with:
* the configuration file with all the run information `config.json`
* the historical records of actions, metrics of interest, optimization and ranking (`.pkl` files)
* the mean and standard deviation of the metrics (`statistics.txt` and `statistics.json`) 

and a  `images` folder contains images both in png and eps format.



