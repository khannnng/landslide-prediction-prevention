# Landslide Prevention and Innovation Challenge

Original competition: https://zindi.africa/competitions/landslide-prevention-and-innovation-challenge. 

I found out about the competition too late to join, but still found the project super interesting and challenging to work on regardless.

## Problem statement
Hong Kong, one of the hilliest and most densely populated cities in the world, is frequently hit by extreme rainfall and is therefore highly susceptible to rain-induced landslides. A landslide is the movement of masses of rock, debris, or earth down a slope and can result in significant loss of life and property. A high-quality landslide inventory is essential not only for landslide hazard and risk analysis but also for supporting agency decisions on landslide hazard mitigation and prevention.

As the common practice is visual, labour-intensive inspection, this hack focuses on automating landslide identification using artificial intelligence techniques and embedding this solution into the creative vision: “Artificial Intelligence for landslide Identification”

The objective of this project is to classify if a landslide occured or not.

Here you can find <a href='https://www.bbc.com/future/article/20220225-how-hong-kong-protects-people-from-its-deadly-landslides '>a great article from BBC</a> why this is an urgent topic!

## Data
Each sample is composed of data from 25 cells, covering an area of 625 m2. Each cell represents an area of 5 x 5 m2 and has nine features. For a landslide sample, cell 13 is the location of landslide, and other cells are the neighboring areas. For a non-landslide sample, there is no recorded landslide occurrence within the sample area.

![image](https://user-images.githubusercontent.com/112837341/232376292-927b35ca-2465-40a2-8694-96fc7e8eadb0.png)


## Main challenge:
The data poses several challenges, including high dimensionality, multicollinearity, latent variables, and data imbalance.

Additionally, domain-specific knowledge of terms such as plan curvation, profile curvation, twi, and lsfactor is necessary for effective analysis. 

## Approach
### Data processing
Derive more summarized or representative metrics from the 25 cell data, such as:
- differences in elevation/slope of the center and surrounding cells, 
- range of elevation/slope of each observation, and the direction of the slope. 
- are the cells facing the in the same general direction or vastly different, and how does that affect the probablity of landslide to occur
- Besides the center cell, the cells directly below it (in term of elevation/slope/aspect) usually have a major role. But how to determine which cell ID are below the center cell is a challenge in itself, since they are different  among each observation and have to be calculated from the aspect (direction of the slope)

The derived metrics can and will create multicollinearity, so a good understanding of the factors and intensive EDA is required to cross check the model result, especially feature importance.

### Model building
Start with some baseline classification models on all features, which may potentially be overfitted. 
Feature selection is done with RFE, VIF and weight of evidence to eliminate multicollinearity and select the most predictive variables.
Hyperparameter tuning will be performed on a few selected models, taking into account factors such as model interpretation, performance, computational expense, and model complexity. 

### Evaluation metrics:
Given data imbalance, F1 will be the primary metric used to evaluate model performance, with AUC, precision, and recall also being considered. 

## Current progress
Baseline model:
- Logistic regression and RF has pretty good performance for its simplicity
- Distance base models like SVM and KNN haven't performed as well as expected
- Boosting model like ADA, XGB are the most promising

Tuned models:
Finished tuning logistic regression and RF to eliminate multi collinearity, only significant variables are selected and reduce overfitting but not yet be able to improve the overall F1 score / AUC. Currently Logistic regression AUC = 0.87, RF AUC = 0.88.

## Next step
Proceed to work on boosting models and try with deep learning models.
