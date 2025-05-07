# Tuberculosis Gene Expression Analysis

## Aim
The aim of this project is to analyze gene expression data from the NCBI GEO dataset (GSE5325) to identify differentially expressed genes between tuberculosis (TB) patients and healthy controls. By leveraging machine learning models, the project seeks to classify samples as TB or control, identify key genes contributing to classification, and visualize the results to uncover biological insights.

## Problem Concepts
Tuberculosis is a major global health issue caused by *Mycobacterium tuberculosis*. Gene expression profiling can reveal molecular differences between TB patients and healthy individuals, aiding in biomarker discovery and understanding disease mechanisms. Key challenges include:

- **High-dimensional data**: Gene expression datasets contain thousands of genes (features), requiring dimensionality reduction to focus on biologically relevant signals.
- **Class imbalance and noise**: Biological data often contains noise and missing values, necessitating robust preprocessing.
- **Interpretability**: Identifying which genes are most important for distinguishing TB from controls is critical for biological interpretation.
- **Model selection**: Different machine learning models may perform variably on gene expression data, requiring comparative evaluation.

This project addresses these challenges by downloading and preprocessing the GSE5325 dataset, applying feature selection, training multiple machine learning models, performing differential expression analysis, and generating visualizations to interpret results.

## Flow of Solution
The solution is implemented in the `improved_tb_analysis.py` script and follows these steps:

1. **Data Loading**:
   - Downloads the GSE5325 dataset from NCBI GEO, which contains gene expression data for TB patients and healthy controls.
   - If the download fails, a simulated dataset is used as a fallback to ensure robustness.
   - Outputs a DataFrame of expression values and a Series of labels (TB=1, Control=0).

2. **Data Preprocessing**:
   - Removes rows with missing values to ensure data quality.
   - Transposes the data so samples are rows and genes are columns.
   - Applies log2 transformation to stabilize variance.
   - Filters out low-variance genes to reduce noise and focus on informative features.

3. **Train/Test Split**:
   - Splits the data into training (70%) and testing (30%) sets, ensuring stratified sampling to maintain class balance.

4. **Feature Selection**:
   - Uses ANOVA F-value to select the top 100 most discriminative genes, reducing dimensionality while retaining biologically relevant features.

5. **Model Training and Comparison**:
   - Trains multiple machine learning models: Random Forest, SVM, KNN, Gradient Boosting, Logistic Regression, and Neural Network.
   - Evaluates models using 5-fold cross-validation and test set accuracy.
   - Outputs performance metrics (accuracy, precision, recall, F1-score) for each model.

6. **Feature Importance Analysis**:
   - Extracts feature importances from the Random Forest model to identify the top 20 genes contributing to classification.
   - Prints the top 10 genes with their importance scores.

7. **Differential Expression Analysis**:
   - Calculates log2 fold change and p-values (t-test) for all genes between TB and control groups.
   - Adjusts p-values using the Benjamini-Hochberg method to control the false discovery rate.
   - Sorts genes by absolute fold change and prints the top 5 differentially expressed genes.

8. **Visualizations**:
   - **Confusion Matrix**: Shows classification performance on the test set.
   - **ROC Curve**: Displays the trade-off between true positive and false positive rates.
   - **Volcano Plot**: Visualizes log2 fold change vs. -log10 p-value to highlight significant genes.
   - **Heatmap**: Shows expression patterns of the top 25 differentially expressed genes, with samples colored by group.
   - **PCA Plot**: Projects samples onto the first two principal components to visualize group separation.
   - **Feature Importance Plot**: Displays the top 15 genes by Random Forest importance.
   - **Box Plots**: Compares expression distributions of the top 5 differentially expressed genes between groups.

## Requirements
To run the script, install the required Python packages:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn scipy statsmodels
```

## Usage
1. Ensure the required packages are installed.
2. Run the script:
   ```bash
   python improved_tb_analysis.py
   ```
3. The script will:
   - Download and process the GSE5325 dataset (or use simulated data if the download fails).
   - Perform preprocessing, model training, and analysis.
   - Save visualizations as PNG files in the working directory.
   - Print results, including model performance, top genes, and differentially expressed genes.

## Outputs
- **Console Output**: Dataset details, model performance metrics, top important genes, and top differentially expressed genes.
- **Visualization Files**:
  - `confusion_matrix.png`
  - `roc_curve.png`
  - `volcano_plot.png`
  - `gene_expression_heatmap.png`
  - `pca_plot.png`
  - `feature_importance.png`
  - `top_deg_boxplots.png`

## Notes
- The script is designed to be robust, with a fallback simulated dataset if the GEO download fails.
- The Random Forest model is used for feature importance due to its robust performance and interpretability.
- Visualizations are saved with clear labels and formatting for easy interpretation.
- The code is modular and can be adapted for other GEO datasets by modifying the `geo_id` parameter in the `download_geo_data` function.