# Python Implementation for Vanilla, Ridge, Lasso Linear Regression from scratch
This is the python implementation of 3 linear regression models from scratch. The model has been tested on the "Geographical Original of Music Data Set" in the UCI machine learning Repository.
Here, X contains 1059 observations and 68 feature dimensions. Y can be either the latitude or longitude. I have tested on both Y
# Result
* It was interesting to find out that Regularization **doesn't necessary mean better performance**. There are several cases where the vanilla model outperforms the regularized one. However, regularized method did indeed show **stable performance** compared to vanilla models. They had in general smaller box plots compared to vanilla. Regularized method(Lasso) also **shows more interpretability** in that less important variables are totally ignored. 

* The performance varied greatly depending on how we split the dataset. Especially for latitude testing, the quantiles of vanilla model was noticeably larger than that of others

# Coefficient Plot
**Vanilla**  
![Image1](/fig/coef_norms_vanilla_latitude.png)    
**Ridge**  
![Image2](/fig/coef_norms_ridge_latitude.png)    
**Lasso**  
![Image3](/fig/coef_norms_lasso_latitude.png)    

# Performance Comparison
**Latitude**  
![Image4](/fig/boxplot_comparison_latitude.png)    
**Longitude**  
![Image5](/fig/boxplot_comparison_longitude.png)    

# Lambda Comparison
**Latitude & Ridge**  
![Image6](/fig/lambda_latitude_ridge.png)    
**Latitude & Lasso**  
![Image7](/fig/lambda_latitude_lasso.png)    

**Longitude & Ridge**  
![Image6](/fig/lambda_longitude_ridge.png)    
**Longitude & Lasso**  
![Image7](/fig/lambda_longitude_lasso.png)    



