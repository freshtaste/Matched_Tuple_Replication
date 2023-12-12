Code documentation for Bai et al. (2023) "Inference for Matched Tuples and Fully Blocked Factorial Designs"
============================================================================================================

Execute "main.py" to generate all tables and figures in the paper. Execute "small exanple for sec 4.3/small_example.py" to generate tables and figures for the simulation study in Section 4.3 with a smaller number of Monte Carlo replications.

The simulations were completed using a computer server with 60 CPU cores. The estimated runtime is approximately 20-30 hours. The small example is expected to complete in approximately 2 hours.

------------------------------------------------------------------------------------------------------------

Data Source

1. Branson, Z., Dasgupta, T. and Rubin, D. B. (2016). Improving covariate balance in 2K factorial designs via rerandomization with an application to a New York City Department of Education High School Study. The Annals of Applied Statistics, 10 1958 – 1976. https://projecteuclid.org/journals/annals-of-applied-statistics/volume-10/issue-4/Improving-covariate-balance-in-2K-factorial-designs-via-rerandomization-with/10.1214/16-AOAS959.full?tab=ArticleLinkSupplemental

2. Fafchamps, M., McKenzie, D., Quinn, S. and Woodruff, C. (2014). Microenterprise growth and the flypaper effect: Evidence from a randomized experiment in ghana. Journal of Development Economics, 106 211–226. https://microdata.worldbank.org/index.php/catalog/2249/study-description 

------------------------------------------------------------------------------------------------------------

Package versions

Python version: 4.1.3

R version: 3.6.3  

nbpMatching version: 1.5.1

numpy version: 1.20.1

joblib version: 1.0.1

statsmodels version: 0.12.2

scipy version: 1.6.2

pandas version: 1.2.4

rpy2 version: 3.5.1

------------------------------------------------------------------------------------------------------------

Note: For the simulation study in Section 4.3, we use the "rpy2" package to connect to the R package "nbpMatching". R and nbpMatching should be installed and running properly before executing the Python program. 