Code documentation for Bai et al. (2023) "Inference for Matched Tuples and Fully Blocked Factorial Designs"
============================================================================================================

Description of content

The replication package includes:
1. Python files that contain code for simulation results (Section 4) and the empirical application (Section 5)
2. PDF files that contain generated figures
3. TXT and csv files that contain generated tables
4. Bash file (train.sh) that contain scripts used to submit jobs to computing clusters
5. Folders (FactorialData, GHA_2008_MGFERE_v01_M_Stata8) that contain the data for simulation and empirical application
6. A folder named "small example for sec 4.3" that contains code (small_example.py), a bash file (train.sh) and outputs (pdf, csv and txt files) for the simulation in Section 4.3 with a smaller number of Monte Carlo replications
7. A README file

------------------------------------------------------------------------------------------------------------

Hardware specifications 

32 G RAM, 60 CPU

------------------------------------------------------------------------------------------------------------

Instructions

Execute "main.py" to generate all tables and figures in the paper. Execute "small example for sec 4.3/small_example.py" to generate tables and figures for the simulation study in Section 4.3 with a smaller number of Monte Carlo replications.

The simulations and empirical applictions were completed using a computer server with 60 CPU cores. The total estimated runtime is approximately 20-30 hours. The small example is expected to complete in approximately 1-2 hours.

------------------------------------------------------------------------------------------------------------

Data Source

1. Branson, Z., Dasgupta, T. and Rubin, D. B. (2016). Improving covariate balance in 2K factorial designs via rerandomization with an application to a New York City Department of Education High School Study. The Annals of Applied Statistics, 10 1958 – 1976. https://projecteuclid.org/journals/annals-of-applied-statistics/volume-10/issue-4/Improving-covariate-balance-in-2K-factorial-designs-via-rerandomization-with/10.1214/16-AOAS959.full?tab=ArticleLinkSupplemental

2. Fafchamps, M., McKenzie, D., Quinn, S. and Woodruff, C. (2014). Microenterprise growth and the flypaper effect: Evidence from a randomized experiment in ghana. Journal of Development Economics, 106 211–226. https://microdata.worldbank.org/index.php/catalog/2249/study-description 

------------------------------------------------------------------------------------------------------------

Package versions

Python version: 3.8.8

R version: 4.1.3

nbpMatching version: 1.5.1

numpy version: 1.20.1

joblib version: 1.0.1

statsmodels version: 0.12.2

scipy version: 1.6.2

pandas version: 1.2.4

rpy2 version: 3.5.1

------------------------------------------------------------------------------------------------------------

Note: For the simulation study in Section 4.3, we use the "rpy2" package to connect to the R package "nbpMatching". R and nbpMatching should be installed and running properly before executing the Python program. 


------------------------------------------------------------------------------------------------------------

An example installation on a free trial AWS server (1 CPU and 1GB of RAM):

1. Install miniconda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh

2. Install Python and packages

conda create --name myenv python=3.8.8
conda activate myenv

conda install numpy=1.20.1
conda install joblib=1.0.1
conda install statsmodels=0.12.2
conda install scipy=1.6.2
conda install pandas=1.2.4

3. Install R and nbpMatching

sudo apt update
sudo apt install r-base

install.packages("ggplot2")
install.packages("rmarkdown")
install.packages("Hmisc")

install.packages("nbpMatching")

4. Install rpy2 (R installation is required for installing rpy2)
pip install rpy2==3.5.1