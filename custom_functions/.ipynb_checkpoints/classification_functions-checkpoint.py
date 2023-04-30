import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import ( 
    f1_score, 
    accuracy_score, 
    recall_score, 
    precision_score, 
    confusion_matrix, 
    roc_auc_score, 
    roc_curve,
    plot_confusion_matrix, 
    classification_report, 
    precision_recall_curve,
    make_scorer,
) 

def cf_percent(y_actual, y_pred, heatmap_labels, ax):
    cf = confusion_matrix(y_actual, y_pred) 
    # group_labels = ['TN\n', 'FP\n', 'FN\n', 'TP\n']
    group_labels = [' ']*4
    group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()] 
    group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)] 
    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)] 
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1]) 
    sns.heatmap(cf, annot=box_labels, fmt="", xticklabels=heatmap_labels, yticklabels=heatmap_labels, ax=ax, cbar=False) 
    
    
def cf_train_test(y_train_actual, y_train_predicted, y_test_actual, y_test_predicted): 
    
    fig, [ax1,ax2] = plt.subplots(ncols=2, figsize=(8,4))
    heatmap_labels = ['Non LS', 'Landslide']
    
    cf_percent(y_train_actual, y_train_predicted, heatmap_labels, ax=ax1)
    cf_percent(y_test_actual, y_test_predicted, heatmap_labels, ax=ax2)
    
    ax1.set(ylabel='Actual', xlabel='Predicted', title='Train') 
    ax2.set(ylabel='Actual', xlabel='Predicted', title='Test')
    plt.show()
    
    
def feature_importance_tree(models_list, limit=25):
    ncols = len(models_list)
    fig, ax = plt.subplots(ncols=ncols, sharex=False, sharey=False, figsize=(4*ncols,8))  
    if ncols == 1:
        ax=[ax]
    for (model, model_name),ax in zip(models_list,ax):
        ax.set_title(model_name)
        importance_df = pd.DataFrame(model.feature_importances_, index = X_train.columns, 
                                     columns = ['Importance']).sort_values(by = 'Importance', ascending = False)
        importance_df = importance_df.head(limit)
        sns.barplot(x = importance_df['Importance'], y = importance_df.index, ax=ax)
        ax.set_yticks([])
        
        for index,y in zip(importance_df.index, np.arange(0.25, len(importance_df.index),1)):
            ax.annotate(index, xy=(0.005,y))
    plt.show()