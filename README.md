# Support Integrity Auditor

This repository contains an automated auditing pipeline that identifies priority mismatches in CRM support tickets. It flags instances where the objective characteristics of a ticket conflict with the priority level assigned by a human agent.

The system is designed to operate without a pre-existing labeled dataset. It bootstraps its own supervision signal, trains a lightweight language model, and outputs structured evidence for every flagged ticket.

## Architecture and Methodology

The pipeline runs in three primary stages:

### 1. Self-Supervised Pseudo-Labeling
Since we lack pre-annotated mismatch labels, the system generates its own ground truth. It fuses independent signals to infer the true severity of a ticket. 
* Keyword Rules: Scans for specific escalation or de-escalation patterns.
* Time Anomalies: Compares actual resolution time against expected category SLAs.
* Lexical Density: Evaluates high-risk word concentration alongside customer satisfaction scores.

Fusion Strategy Justification: 
We used a Logistic Regression model to fuse these signals. Based on the generated ablation table, the semantic features and keyword density carried the heaviest coefficient weights. Resolution time served as a strong secondary validation signal, particularly for detecting hidden crises where the ticket took far longer to solve than its assigned low priority would suggest.

### 2. Classifier Training
We use microsoft/deberta-v3-small as the base model. To make training efficient on consumer hardware, we apply Low-Rank Adaptation (LoRA) to the attention modules. 

The dataset is highly imbalanced because most human agents classify tickets correctly. We address this class imbalance by upsampling the minority mismatch class and applying dynamic class weights during the CrossEntropyLoss calculation.

### 3. Inference and Dossier Generation
The trained model runs over the test split and identifies tickets where the assigned priority and inferred severity do not align. For every mismatch, the system generates a structured JSON dossier containing the probability score, the severity delta, and specific textual evidence that triggered the flag.

## Repository Structure

* train_pipeline.py: The standalone script for data processing, pseudo-label generation, and LoRA model training.
* predict.py: The inference script that reads the trained weights, evaluates tickets, and outputs the evidence dossiers.
* app.py: A Streamlit web dashboard for macro analytics and live adversarial ticket testing.
* notebook.ipynb: The complete end-to-end reproducible pipeline, including Matplotlib and Seaborn evaluations (ROC curve, AUC, confusion matrix).
* requirements.txt: All pinned dependencies for the project.

## Setup Instructions

1. Clone the repository and navigate to the project directory.
2. Install the required dependencies:
pip install -r requirements.txt

3. To train the model from scratch, ensure your raw dataset is in the root folder and run:
python train_pipeline.py

4. To generate predictions and the JSON dossiers:
python predict.py

5. To launch the interactive dashboard locally:
streamlit run app.py
