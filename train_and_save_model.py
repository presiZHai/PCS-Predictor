import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

def train_and_save():
    """
    This function performs the full pipeline:
    1. Loads the raw data ('clients_data.csv').
    2. Preprocesses the data, including creating PCA components.
    3. Trains a RandomForestClassifier on the 12 PCA components.
    4. Saves the trained model and the list of PCA feature names.
    """
    print("--- Starting model training and saving process ---")

    # --- 1. Load Data ---
    try:
        clients_df = pd.read_csv('clients_data.csv')
        print("Successfully loaded 'clients_data.csv'.")
    except FileNotFoundError:
        print("Error: 'clients_data.csv' not found. Please place it in the same directory.")
        return

    # Fix column naming inconsistency
    clients_df.rename(columns={'Lab Tests Explained': 'LabTests_Explained'}, inplace=True)

    # --- 2. Preprocessing ---
    # A. Initial transformations
    age_mapping = {'18-24': 21, '25-34': 29.5, '35-44': 39.5, '45-54': 49.5, '55 years and above': 65}
    clients_df['Age'] = clients_df['Age'].replace(age_mapping)

    children_mapping = {'1-2': 1.5, '3-4': 3.5, 'Greater than 4': 5, 'None': 0}
    clients_df['Num of Children'] = clients_df['Num of Children'].replace(children_mapping)
    clients_df['Num of Children'] = clients_df['Num of Children'].infer_objects(copy=False)

    clients_df['Monthly Income'] = clients_df['Monthly Income'].astype(str).str.replace('–', '-', regex=False)

    income_mapping = {
        'Less than 20,000 Naira': 10000,
        '20,000-50,000 Naira': 35000,
        '51,000-100,000 Naira': 75500,
        '101,000-200,000 Naira': 150500,
        'More than 200,000 Naira': 250000,
        'Prefer not to say': np.nan
    }
    clients_df['Monthly Income'] = clients_df['Monthly Income'].replace(income_mapping)
    clients_df['Monthly Income'] = pd.to_numeric(clients_df['Monthly Income'], errors='coerce')
    clients_df['Monthly Income'] = clients_df['Monthly Income'].fillna(clients_df['Monthly Income'].median())

    # B. PCC Questions to Numeric
    pcc_questions_cols = [
        'Greet_Comfort', 'Discuss_VisitReason', 'Encourage_Thoughts', 'Listen_Careful',
        'Understood_You', 'Exam_Explained', 'LabTests_Explained',
        'Discuss_TreatOptions', 'Info_AsDesired', 'Plan_Acceptability_Check',
        'Meds_Explained_SideFX', 'Encourage_Questions', 'Respond_Q_Concerns',
        'Showed_Personal_Concern', 'Involved_In_Decisions', 'Discuss_NextSteps',
        'Checked_Understanding', 'Time_Spent_Adequate'
    ]
    likert_mapping_pcc = {
        'Strongly Disagree': 1,
        'Disagree': 2,
        'Neither Agree or Disagree': 3,
        'Agree': 4,
        'Strongly Agree': 5
    }
    for col in pcc_questions_cols:
        clients_df[col] = clients_df[col].replace(likert_mapping_pcc)
        clients_df[col] = pd.to_numeric(clients_df[col], errors='coerce')
        clients_df[col] = clients_df[col].fillna(clients_df[col].median())
        clients_df[col] = clients_df[col].infer_objects(copy=False)

    # C. Target Variable
    satisfaction_mapping = {
        'Very dissatisfied': 0,
        'Neutral': 1,
        'Satisfied': 2,
        'Very satisfied': 3
    }
    clients_df['Satisfaction'] = clients_df['Visit_Satisfaction'].map(satisfaction_mapping)
    clients_df.dropna(subset=['Satisfaction'], inplace=True)
    clients_df['Satisfaction'] = clients_df['Satisfaction'].astype(int)

    # --- 3. PCA on PCC Questions ---
    pcc_data = clients_df[pcc_questions_cols].copy()
    scaler = StandardScaler()
    pcc_scaled = scaler.fit_transform(pcc_data)

    pca = PCA(n_components=12)
    pcc_components = pca.fit_transform(pcc_scaled)

    interpreted_pc_names = {
        'PC1': 'Overall Communication & Rapport',
        'PC2': 'Information & Explanation Focus',
        'PC3': 'Initial Rapport & Comfort',
        'PC4': 'Adequate Time Spent',
        'PC5': 'Thoroughness & Follow-up',
        'PC6': 'Patient Understanding & Engagement',
        'PC7': 'Treatment Discussion & Personal Care',
        'PC8': 'Initial Dialogue & Encouragement',
        'PC9': 'Medical Explanation & Involvement in Decisions',
        'PC10': 'Confirmation of Understanding',
        'PC11': 'Information Seeking vs. Questioning',
        'PC12': 'Treatment Option Management'
    }

    pca_df = pd.DataFrame(data=pcc_components, columns=[f'PC{i+1}' for i in range(12)])
    pca_df.rename(columns=interpreted_pc_names, inplace=True)

    # --- 4. Prepare Final Training Data ---
    X = pca_df
    y = clients_df['Satisfaction'].loc[X.index].copy()
    feature_names = list(X.columns)
    print(f"Model will be trained on these {len(feature_names)} features:\n{feature_names}")

    # --- 5. Train Model ---
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
    class_weights_dict = dict(zip(np.unique(y), class_weights))

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weights_dict)
    model.fit(X, y)
    print("Model training complete.")

    # --- 6. Save Artifacts ---
    joblib.dump(model, 'final_model.joblib')
    joblib.dump(feature_names, 'feature_names.joblib')
    print("Successfully saved 'final_model.joblib' and 'feature_names.joblib'.")
    print("--- Process complete ---")


if __name__ == '__main__':
    train_and_save()