import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

class StudentDropoutPredictor:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.original_df = None
        
        # Create directory for saving plots
        self.output_dir = 'dropout_analysis_plots'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
    def load_and_preprocess_data(self, file_path):
        """
        Load and preprocess the UCI Student Dropout dataset with proper column handling
        """
        print(f"Loading dataset from: {file_path}")
        
        try:
            # Try different separators commonly used in datasets
            try:
                df = pd.read_csv(file_path, sep=';')
                print("Dataset loaded with semicolon separator")
            except:
                try:
                    df = pd.read_csv(file_path, sep=',')
                    print("Dataset loaded with comma separator")
                except:
                    df = pd.read_csv(file_path, sep='\t')
                    print("Dataset loaded with tab separator")
            
            print(f"Raw dataset shape: {df.shape}")
            print(f"Columns found: {list(df.columns)}")
            
            # Clean column names - remove extra spaces and standardize
            df.columns = df.columns.str.strip()
            
            # Create a mapping for expected column names
            column_mapping = {
                'Marital status': 'Marital_status',
                'Application mode': 'Application_mode',
                'Application order': 'Application_order',
                'Course': 'Course',
                'Daytime/evening attendance\t': 'Daytime_evening_attendance',
                'Previous qualification': 'Previous_qualification',
                'Previous qualification (grade)': 'Previous_qualification_grade',
                'Nacionality': 'Nationality',
                "Mother's qualification": 'Mothers_qualification',
                "Father's qualification": 'Fathers_qualification',
                "Mother's occupation": 'Mothers_occupation',
                "Father's occupation": 'Fathers_occupation',
                'Admission grade': 'Admission_grade',
                'Displaced': 'Displaced',
                'Educational special needs': 'Educational_special_needs',
                'Debtor': 'Debtor',
                'Tuition fees up to date': 'Tuition_fees_up_to_date',
                'Gender': 'Gender',
                'Scholarship holder': 'Scholarship_holder',
                'Age at enrollment': 'Age_at_enrollment',
                'International': 'International',
                'Curricular units 1st sem (credited)': 'Curricular_units_1st_sem_credited',
                'Curricular units 1st sem (enrolled)': 'Curricular_units_1st_sem_enrolled',
                'Curricular units 1st sem (evaluations)': 'Curricular_units_1st_sem_evaluations',
                'Curricular units 1st sem (approved)': 'Curricular_units_1st_sem_approved',
                'Curricular units 1st sem (grade)': 'Curricular_units_1st_sem_grade',
                'Curricular units 1st sem (without evaluations)': 'Curricular_units_1st_sem_without_evaluations',
                'Curricular units 2nd sem (credited)': 'Curricular_units_2nd_sem_credited',
                'Curricular units 2nd sem (enrolled)': 'Curricular_units_2nd_sem_enrolled',
                'Curricular units 2nd sem (evaluations)': 'Curricular_units_2nd_sem_evaluations',
                'Curricular units 2nd sem (approved)': 'Curricular_units_2nd_sem_approved',
                'Curricular units 2nd sem (grade)': 'Curricular_units_2nd_sem_grade',
                'Curricular units 2nd sem (without evaluations)': 'Curricular_units_2nd_sem_without_evaluations',
                'Unemployment rate': 'Unemployment_rate',
                'Inflation rate': 'Inflation_rate',
                'GDP': 'GDP',
                'Target': 'Target'
            }
            
            # Rename columns
            df = df.rename(columns=column_mapping)
            print("Column names standardized")
            
            # Check for missing values
            missing_values = df.isnull().sum()
            if missing_values.sum() > 0:
                print(f"\nMissing values found:")
                print(missing_values[missing_values > 0])
                
                # Handle missing values
                # For numerical columns: fill with median
                numerical_cols = df.select_dtypes(include=[np.number]).columns
                df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
                
                # For categorical columns: fill with mode
                categorical_cols = df.select_dtypes(include=['object']).columns
                for col in categorical_cols:
                    df[col] = df[col].fillna(df[col].mode()[0])
                
                print("Missing values handled")
            
            # Check if Target column exists
            if 'Target' not in df.columns:
                raise ValueError("Target column not found in dataset. Expected column name: 'Target'")
            
            print(f"Dataset loaded successfully with {len(df)} samples and {len(df.columns)} features")
            print(f"Target distribution:\n{df['Target'].value_counts()}")
            
            # Store original dataframe for visualization
            self.original_df = df.copy()
            
            # Display first few rows for verification
            print(f"\nFirst 5 rows preview:")
            print(df.head())
            
            return df
            
        except FileNotFoundError:
            raise FileNotFoundError(f"""
            Dataset file not found: {file_path}
            
            To use this model, please download the UCI Student Dropout dataset:
            1. Visit: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
            2. Download the dataset file (usually named 'data.csv' or similar)
            3. Place it in your project directory
            4. Update the file_path parameter when calling this function
            """)
        
        except Exception as e:
            raise Exception(f"""
            Error loading dataset: {str(e)}
            
            Please ensure:
            1. File path is correct
            2. Dataset is in CSV format
            3. Dataset contains a 'Target' column with dropout/success labels
            4. Column names match expected format (see UCI dataset documentation)
            """)
    
    def create_exploratory_visualizations(self, df):
        """
        Create comprehensive exploratory data analysis visualizations
        """
        print("Creating exploratory data analysis visualizations...")
        
        # Set style for better plots
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Target Distribution
        plt.figure(figsize=(10, 6))
        target_counts = df['Target'].value_counts()
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        
        plt.subplot(1, 2, 1)
        bars = plt.bar(target_counts.index, target_counts.values, color=colors)
        plt.title('Distribution of Student Outcomes', fontsize=14, fontweight='bold')
        plt.xlabel('Outcome')
        plt.ylabel('Number of Students')
        for bar, count in zip(bars, target_counts.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, 
                    f'{count}\n({count/len(df)*100:.1f}%)', 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.subplot(1, 2, 2)
        plt.pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        plt.title('Student Outcome Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/01_target_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Age Distribution by Outcome
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        for target in df['Target'].unique():
            subset = df[df['Target'] == target]
            plt.hist(subset['Age_at_enrollment'], alpha=0.7, label=target, bins=20)
        plt.xlabel('Age at Enrollment')
        plt.ylabel('Frequency')
        plt.title('Age Distribution by Outcome')
        plt.legend()
        
        # 3. Academic Performance Analysis
        plt.subplot(2, 2, 2)
        sns.boxplot(data=df, x='Target', y='Previous_qualification_grade')
        plt.title('Previous Qualification Grade by Outcome')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 3)
        sns.boxplot(data=df, x='Target', y='Admission_grade')
        plt.title('Admission Grade by Outcome')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 4)
        sns.boxplot(data=df, x='Target', y='Curricular_units_1st_sem_grade')
        plt.title('1st Semester Grade by Outcome')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/02_academic_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. Financial Factors Analysis
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        tuition_dropout = pd.crosstab(df['Tuition_fees_up_to_date'], df['Target'], normalize='index') * 100
        tuition_dropout.plot(kind='bar', ax=plt.gca())
        plt.title('Dropout Rate by Tuition Payment Status')
        plt.xlabel('Tuition Fees Up to Date (0=No, 1=Yes)')
        plt.ylabel('Percentage')
        plt.legend(title='Outcome')
        plt.xticks(rotation=0)
        
        plt.subplot(1, 3, 2)
        debtor_dropout = pd.crosstab(df['Debtor'], df['Target'], normalize='index') * 100
        debtor_dropout.plot(kind='bar', ax=plt.gca())
        plt.title('Dropout Rate by Debtor Status')
        plt.xlabel('Debtor (0=No, 1=Yes)')
        plt.ylabel('Percentage')
        plt.legend(title='Outcome')
        plt.xticks(rotation=0)
        
        plt.subplot(1, 3, 3)
        scholarship_dropout = pd.crosstab(df['Scholarship_holder'], df['Target'], normalize='index') * 100
        scholarship_dropout.plot(kind='bar', ax=plt.gca())
        plt.title('Dropout Rate by Scholarship Status')
        plt.xlabel('Scholarship Holder (0=No, 1=Yes)')
        plt.ylabel('Percentage')
        plt.legend(title='Outcome')
        plt.xticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/03_financial_factors_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 5. Parent Education Impact
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        mother_edu_dropout = df.groupby('Mothers_qualification')['Target'].apply(
            lambda x: (x == 'Dropout').sum() / len(x) * 100
        )
        mother_edu_dropout.plot(kind='bar')
        plt.title('Dropout Rate by Mother\'s Education Level')
        plt.xlabel('Mother\'s Qualification Level')
        plt.ylabel('Dropout Rate (%)')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        father_edu_dropout = df.groupby('Fathers_qualification')['Target'].apply(
            lambda x: (x == 'Dropout').sum() / len(x) * 100
        )
        father_edu_dropout.plot(kind='bar')
        plt.title('Dropout Rate by Father\'s Education Level')
        plt.xlabel('Father\'s Qualification Level')
        plt.ylabel('Dropout Rate (%)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/04_parent_education_impact.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Exploratory visualizations saved to {self.output_dir}/")
    
    def feature_engineering(self, df):
        """
        Create additional features relevant to university context
        """
        print("Creating engineered features...")
        
        # Academic Performance Indicators
        df['Total_1st_sem_units'] = df['Curricular_units_1st_sem_enrolled']
        df['Total_2nd_sem_units'] = df['Curricular_units_2nd_sem_enrolled']
        df['Success_rate_1st_sem'] = df['Curricular_units_1st_sem_approved'] / (df['Curricular_units_1st_sem_enrolled'] + 0.1)
        df['Success_rate_2nd_sem'] = df['Curricular_units_2nd_sem_approved'] / (df['Curricular_units_2nd_sem_enrolled'] + 0.1)
        df['Overall_grade_average'] = (df['Curricular_units_1st_sem_grade'] + df['Curricular_units_2nd_sem_grade']) / 2
        
        # Financial Stress Indicators
        df['Financial_stress'] = ((df['Tuition_fees_up_to_date'] == 0) | (df['Debtor'] == 1)).astype(int)
        df['Economic_pressure'] = (df['Unemployment_rate'] > df['Unemployment_rate'].median()).astype(int)
        
        # Family Support Indicators
        df['Parent_education_support'] = df['Fathers_qualification'] + df['Mothers_qualification']
        df['Family_stability'] = ((df['Displaced'] == 0) & (df['Educational_special_needs'] == 0)).astype(int)
        
        # Age-related factors
        df['Mature_student'] = (df['Age_at_enrollment'] > 22).astype(int)
        df['Age_risk'] = (df['Age_at_enrollment'] > 25).astype(int)
        
        print(f"Added 11 engineered features")
        return df
    
    def prepare_features(self, df):
        """
        Prepare features for machine learning
        """
        # Separate features and target
        X = df.drop('Target', axis=1)
        y = df['Target']
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Handle any categorical variables (if present)
        categorical_features = X.select_dtypes(include=['object']).columns
        for col in categorical_features:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            self.label_encoders[col] = le
        
        # Convert target to binary (Dropout vs Non-Dropout)
        y_binary = (y == 'Dropout').astype(int)
        
        return X, y, y_binary
    
    def train_models(self, X, y_binary):
        """
        Train multiple models and compare performance
        """
        print("Training multiple models...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Store test data for evaluation
        self.X_test = X_test
        self.y_test = y_test
        self.X_test_scaled = X_test_scaled
        
        # Define models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(probability=True, random_state=42)
        }
        
        # Train and evaluate each model
        results = {}
        for name, model in models.items():
            print(f"Training {name}...")
            
            if name in ['Logistic Regression', 'SVM']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'auc_score': auc_score,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            self.models[name] = model
            
            print(f"{name} - Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}")
        
        # Select best model based on AUC score
        best_model_name = max(results.keys(), key=lambda x: results[x]['auc_score'])
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name
        
        print(f"\nBest model: {best_model_name} (AUC: {results[best_model_name]['auc_score']:.4f})")
        
        return results
    
    def create_model_performance_visualizations(self, results):
        """
        Create visualizations for model performance comparison
        """
        print("Creating model performance visualizations...")
        
        # 1. Model Comparison
        plt.figure(figsize=(15, 5))
        
        models = list(results.keys())
        accuracies = [results[model]['accuracy'] for model in models]
        auc_scores = [results[model]['auc_score'] for model in models]
        
        plt.subplot(1, 3, 1)
        bars = plt.bar(models, accuracies, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
        plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.subplot(1, 3, 2)
        bars = plt.bar(models, auc_scores, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
        plt.title('Model AUC Score Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('AUC Score')
        plt.xticks(rotation=45)
        for bar, auc in zip(bars, auc_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{auc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. ROC Curves
        plt.subplot(1, 3, 3)
        for model_name in models:
            fpr, tpr, _ = roc_curve(self.y_test, results[model_name]['probabilities'])
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {results[model_name]["auc_score"]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/05_model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Best Model Confusion Matrix
        plt.figure(figsize=(10, 4))
        
        best_result = results[self.best_model_name]
        y_pred = best_result['predictions']
        
        plt.subplot(1, 2, 1)
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Stay', 'Dropout'], yticklabels=['Stay', 'Dropout'])
        plt.title(f'Confusion Matrix - {self.best_model_name}', fontweight='bold')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        # Normalized confusion matrix
        plt.subplot(1, 2, 2)
        cm_norm = confusion_matrix(self.y_test, y_pred, normalize='true')
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                   xticklabels=['Stay', 'Dropout'], yticklabels=['Stay', 'Dropout'])
        plt.title(f'Normalized Confusion Matrix - {self.best_model_name}', fontweight='bold')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/06_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_feature_importance_visualization(self):
        """
        Create feature importance visualization
        """
        if hasattr(self.best_model, 'feature_importances_'):
            print("Creating feature importance visualization...")
            
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            plt.figure(figsize=(12, 10))
            top_features = feature_importance.tail(20)
            
            bars = plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'Top 20 Most Important Features - {self.best_model_name}', 
                     fontsize=14, fontweight='bold')
            
            # Add value labels on bars
            for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
                plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                        f'{importance:.3f}', va='center', ha='left', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/07_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            return feature_importance
        else:
            print("Feature importance not available for this model type")
            return None
    
    def create_risk_analysis_visualizations(self, X, y_binary):
        """
        Create risk analysis and prediction visualizations
        """
        print("Creating risk analysis visualizations...")
        
        # Get predictions for all data
        if self.best_model_name in ['Logistic Regression', 'SVM']:
            X_scaled = self.scaler.transform(X)
            probabilities = self.best_model.predict_proba(X_scaled)[:, 1]
        else:
            probabilities = self.best_model.predict_proba(X)[:, 1]
        
        # Create risk categories
        risk_categories = pd.cut(probabilities, 
                               bins=[0, 0.3, 0.5, 0.7, 1.0], 
                               labels=['Low', 'Medium', 'High', 'Critical'])
        
        # 1. Risk Distribution
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        risk_counts = risk_categories.value_counts()
        colors = ['#2ecc71', '#f39c12', '#e74c3c', '#8e44ad']
        bars = plt.bar(risk_counts.index, risk_counts.values, color=colors)
        plt.title('Distribution of Students by Risk Level', fontsize=14, fontweight='bold')
        plt.xlabel('Risk Level')
        plt.ylabel('Number of Students')
        for bar, count in zip(bars, risk_counts.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                    f'{count}\n({count/len(probabilities)*100:.1f}%)', 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.subplot(1, 3, 2)
        plt.hist(probabilities, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(probabilities.mean(), color='red', linestyle='--', 
                   label=f'Mean: {probabilities.mean():.3f}')
        plt.xlabel('Dropout Probability')
        plt.ylabel('Number of Students')
        plt.title('Distribution of Dropout Probabilities', fontsize=14, fontweight='bold')
        plt.legend()
        
        plt.subplot(1, 3, 3)
        risk_accuracy = []
        risk_labels = ['Low', 'Medium', 'High', 'Critical']
        for risk_level in risk_labels:
            mask = risk_categories == risk_level
            if mask.sum() > 0:
                actual_dropout_rate = y_binary[mask].mean()
                risk_accuracy.append(actual_dropout_rate * 100)
            else:
                risk_accuracy.append(0)
        
        bars = plt.bar(risk_labels, risk_accuracy, color=colors)
        plt.title('Actual Dropout Rate by Risk Level', fontsize=14, fontweight='bold')
        plt.xlabel('Risk Level')
        plt.ylabel('Actual Dropout Rate (%)')
        for bar, rate in zip(bars, risk_accuracy):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/08_risk_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. High-Risk Student Characteristics
        high_risk_mask = probabilities >= 0.7
        high_risk_students = self.original_df[high_risk_mask]
        
        if len(high_risk_students) > 0:
            plt.figure(figsize=(15, 10))
            
            # Age distribution of high-risk students
            plt.subplot(2, 3, 1)
            plt.hist(high_risk_students['Age_at_enrollment'], bins=20, alpha=0.7, color='red')
            plt.xlabel('Age at Enrollment')
            plt.ylabel('Count')
            plt.title('Age Distribution - High Risk Students')
            
            # Academic performance of high-risk students
            plt.subplot(2, 3, 2)
            plt.hist(high_risk_students['Previous_qualification_grade'], bins=20, alpha=0.7, color='red')
            plt.xlabel('Previous Qualification Grade')
            plt.ylabel('Count')
            plt.title('Previous Grade Distribution - High Risk')
            
            plt.subplot(2, 3, 3)
            plt.hist(high_risk_students['Curricular_units_1st_sem_grade'], bins=20, alpha=0.7, color='red')
            plt.xlabel('1st Semester Grade')
            plt.ylabel('Count')
            plt.title('1st Sem Grade Distribution - High Risk')
            
            # Financial factors
            plt.subplot(2, 3, 4)
            tuition_status = high_risk_students['Tuition_fees_up_to_date'].value_counts()
            plt.pie(tuition_status.values, labels=['Behind on Fees', 'Up to Date'], 
                   autopct='%1.1f%%', colors=['red', 'lightgreen'])
            plt.title('Tuition Payment Status - High Risk')
            
            plt.subplot(2, 3, 5)
            scholarship_status = high_risk_students['Scholarship_holder'].value_calls()
            plt.pie(scholarship_status.values, labels=['No Scholarship', 'Has Scholarship'], 
                   autopct='%1.1f%%', colors=['red', 'lightblue'])
            plt.title('Scholarship Status - High Risk')
            
            plt.subplot(2, 3, 6)
            debtor_status = high_risk_students['Debtor'].value_counts()
            plt.pie(debtor_status.values, labels=['Not Debtor', 'Debtor'], 
                   autopct='%1.1f%%', colors=['lightgreen', 'red'])
            plt.title('Debtor Status - High Risk')
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/09_high_risk_student_characteristics.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        return probabilities, risk_categories
    
    def create_intervention_recommendations(self, probabilities, risk_categories):
        """
        Create visualizations for intervention recommendations
        """
        print("Creating intervention recommendation visualizations...")
        
        # Count students by risk level
        risk_counts = risk_categories.value_counts()
        
        # Create intervention priority matrix
        plt.figure(figsize=(15, 10))
        
        # Priority matrix
        plt.subplot(2, 2, 1)
        intervention_data = {
            'Risk Level': ['Critical', 'High', 'Medium', 'Low'],
            'Students': [risk_counts.get('Critical', 0), risk_counts.get('High', 0), 
                        risk_counts.get('Medium', 0), risk_counts.get('Low', 0)],
            'Priority': ['Immediate', 'Urgent', 'Monitor', 'Preventive'],
            'Colors': ['#8e44ad', '#e74c3c', '#f39c12', '#2ecc71']
        }
        
        bars = plt.bar(intervention_data['Risk Level'], intervention_data['Students'], 
                      color=intervention_data['Colors'])
        plt.title('Intervention Priority by Risk Level', fontsize=14, fontweight='bold')
        plt.xlabel('Risk Level')
        plt.ylabel('Number of Students')
        
        for bar, count, priority in zip(bars, intervention_data['Students'], intervention_data['Priority']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                    f'{count}\n{priority}', ha='center', va='bottom', fontweight='bold')
        
        # Intervention strategies
        plt.subplot(2, 2, 2)
        strategies = ['Academic Support', 'Financial Aid', 'Counseling', 'Mentoring', 'Study Groups']
        effectiveness = [85, 70, 65, 60, 55]  # Example effectiveness percentages
        
        bars = plt.barh(strategies, effectiveness, color=['#3498db', '#e67e22', '#9b59b6', '#1abc9c', '#34495e'])
        plt.xlabel('Effectiveness (%)')
        plt.title('Intervention Strategy Effectiveness', fontsize=14, fontweight='bold')
        
        for bar, eff in zip(bars, effectiveness):
            plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                    f'{eff}%', va='center', ha='left', fontweight='bold')
        
        # Cost-benefit analysis
        plt.subplot(2, 2, 3)
        interventions = ['Early Warning\nSystem', 'Academic\nTutoring', 'Financial\nCounseling', 
                        'Mental Health\nSupport', 'Career\nGuidance']
        costs = [20, 150, 80, 120, 60]  # Cost per student
        benefits = [300, 400, 250, 350, 200]  # Benefit per student retained
        
        x = range(len(interventions))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], costs, width, label='Cost per Student', color='lightcoral')
        plt.bar([i + width/2 for i in x], benefits, width, label='Benefit per Student', color='lightgreen')
        
        plt.xlabel('Intervention Type')
        plt.ylabel('Amount ($)')
        plt.title('Cost-Benefit Analysis of Interventions')
        plt.xticks(x, interventions, rotation=45)
        plt.legend()
        
        # Resource allocation
        plt.subplot(2, 2, 4)
        resource_allocation = {
            'Critical (Immediate)': risk_counts.get('Critical', 0) * 200,  # $200 per critical student
            'High (Urgent)': risk_counts.get('High', 0) * 150,  # $150 per high-risk student
            'Medium (Monitor)': risk_counts.get('Medium', 0) * 75,  # $75 per medium-risk student
            'Low (Preventive)': risk_counts.get('Low', 0) * 25   # $25 per low-risk student
        }
        
        plt.pie(resource_allocation.values(), labels=resource_allocation.keys(), 
               autopct='%1.1f%%', colors=['#8e44ad', '#e74c3c', '#f39c12', '#2ecc71'])
        plt.title('Recommended Resource Allocation')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/10_intervention_recommendations.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create detailed intervention report
        self.create_intervention_report(risk_counts)
    
    def create_intervention_report(self, risk_counts):
        """
        Create a detailed intervention report
        """
        total_students = sum(risk_counts.values())
        critical_students = risk_counts.get('Critical', 0)
        high_risk_students = risk_counts.get('High', 0)
        
        report = f"""
        STUDENT DROPOUT RISK ANALYSIS REPORT
        =====================================
        Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        EXECUTIVE SUMMARY
        -----------------
        Total Students Analyzed: {total_students:,}
        
        Risk Distribution:
        - Critical Risk (≥70% dropout probability): {risk_counts.get('Critical', 0):,} students ({risk_counts.get('Critical', 0)/total_students*100:.1f}%)
        - High Risk (50-69% dropout probability): {risk_counts.get('High', 0):,} students ({risk_counts.get('High', 0)/total_students*100:.1f}%)
        - Medium Risk (30-49% dropout probability): {risk_counts.get('Medium', 0):,} students ({risk_counts.get('Medium', 0)/total_students*100:.1f}%)
        - Low Risk (<30% dropout probability): {risk_counts.get('Low', 0):,} students ({risk_counts.get('Low', 0)/total_students*100:.1f}%)
        
        IMMEDIATE ACTION REQUIRED
        -------------------------
        {critical_students + high_risk_students:,} students require immediate intervention
        
        RECOMMENDED INTERVENTIONS
        -------------------------
        
        1. CRITICAL RISK STUDENTS ({critical_students:,} students)
           - Immediate one-on-one counseling sessions
           - Academic support and tutoring
           - Financial aid assessment and assistance
           - Weekly progress monitoring
           - Family engagement programs
           - Estimated cost: ${critical_students * 200:,}
        
        2. HIGH RISK STUDENTS ({high_risk_students:,} students)
           - Group counseling sessions
           - Study skills workshops
           - Peer mentoring programs
           - Monthly progress check-ins
           - Financial literacy programs
           - Estimated cost: ${high_risk_students * 150:,}
        
        3. MEDIUM RISK STUDENTS ({risk_counts.get('Medium', 0):,} students)
           - Early warning system alerts
           - Study group participation
           - Career guidance sessions
           - Semester progress reviews
           - Estimated cost: ${risk_counts.get('Medium', 0) * 75:,}
        
        4. LOW RISK STUDENTS ({risk_counts.get('Low', 0):,} students)
           - General wellness programs
           - Leadership development opportunities
           - Peer mentoring roles
           - Estimated cost: ${risk_counts.get('Low', 0) * 25:,}
        
        TOTAL ESTIMATED BUDGET REQUIRED
        -------------------------------
        ${critical_students * 200 + high_risk_students * 150 + risk_counts.get('Medium', 0) * 75 + risk_counts.get('Low', 0) * 25:,}
        
        KEY PERFORMANCE INDICATORS (KPIs)
        ----------------------------------
        - Reduction in dropout rate by 25% within one academic year
        - Increase in student retention rate to 85%
        - Improvement in academic performance metrics
        - Student satisfaction scores above 4.0/5.0
        
        IMPLEMENTATION TIMELINE
        -----------------------
        Week 1-2: Identify and contact critical and high-risk students
        Week 3-4: Begin intensive interventions for critical risk students
        Month 2: Implement support programs for high-risk students
        Month 3: Launch monitoring systems for medium-risk students
        Ongoing: Preventive programs for all students
        
        """
        
        # Save report to file
        with open(f'{self.output_dir}/intervention_report.txt', 'w') as f:
            f.write(report)
        
        print("Detailed intervention report saved to intervention_report.txt")
        print(report)
    
    def evaluate_model(self, results):
        """
        Detailed evaluation of the best model
        """
        print(f"\n=== Detailed Evaluation: {self.best_model_name} ===")
        
        best_result = results[self.best_model_name]
        y_pred = best_result['predictions']
        y_pred_proba = best_result['probabilities']
        
        # Classification Report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, 
                                  target_names=['Non-Dropout', 'Dropout']))
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"True Negative: {cm[0,0]}, False Positive: {cm[0,1]}")
        print(f"False Negative: {cm[1,0]}, True Positive: {cm[1,1]}")
        
        # Feature Importance (if available)
        feature_importance = self.create_feature_importance_visualization()
        
        if feature_importance is not None:
            print(f"\nTop 10 Most Important Features:")
            for idx, row in feature_importance.tail(10).iterrows():
                print(f"{row['feature']}: {row['importance']:.4f}")
        
        return best_result
    
    def predict_dropout_risk(self, student_data):
        """
        Predict dropout risk for a single student
        """
        if self.best_model is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        # Convert to DataFrame if needed
        if isinstance(student_data, dict):
            student_df = pd.DataFrame([student_data])
        else:
            student_df = student_data.copy()
        
        # Apply same preprocessing
        for col in self.label_encoders:
            if col in student_df.columns:
                student_df[col] = self.label_encoders[col].transform(student_df[col])
        
        # Scale features if needed
        if self.best_model_name in ['Logistic Regression', 'SVM']:
            student_features = self.scaler.transform(student_df)
        else:
            student_features = student_df
        
        # Make prediction
        dropout_probability = self.best_model.predict_proba(student_features)[0, 1]
        prediction = self.best_model.predict(student_features)[0]
        
        # Risk categories
        if dropout_probability >= 0.7:
            risk_level = "Critical"
        elif dropout_probability >= 0.5:
            risk_level = "High"
        elif dropout_probability >= 0.3:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        return {
            'dropout_probability': round(dropout_probability * 100, 2),
            'prediction': 'Dropout' if prediction == 1 else 'Stay',
            'risk_level': risk_level,
            'model_used': self.best_model_name
        }

# Example usage and demonstration
if __name__ == "__main__":
    print("=== Student Dropout Prediction Model with Comprehensive Analysis ===\n")
    
    # Initialize predictor
    predictor = StudentDropoutPredictor()
    
    # Load and preprocess data from actual file
    dataset_file = r'C:/Users/OSL/Downloads/Dropout Prediction/data.csv'  
    
    try:
        df = predictor.load_and_preprocess_data(dataset_file)
        
        # Create exploratory visualizations
        predictor.create_exploratory_visualizations(df)
        
        # Feature engineering
        df = predictor.feature_engineering(df)
        
        # Prepare features
        X, y, y_binary = predictor.prepare_features(df)
        
        print(f"\nFinal dataset shape: {X.shape}")
        print(f"Dropout rate: {y_binary.mean():.2%}")
        
        # Train models
        results = predictor.train_models(X, y_binary)
        
        # Create model performance visualizations
        predictor.create_model_performance_visualizations(results)
        
        # Evaluate best model
        best_result = predictor.evaluate_model(results)
        
        # Create risk analysis visualizations
        probabilities, risk_categories = predictor.create_risk_analysis_visualizations(X, y_binary)
        
        # Create intervention recommendations
        predictor.create_intervention_recommendations(probabilities, risk_categories)
        
        print("\n" + "="*50)
        print("MAKING PREDICTIONS FOR SAMPLE STUDENTS")
        print("="*50)
        
        # Example predictions for different risk profiles
        sample_students = [
            {
                'Previous_qualification_grade': 15.5,
                'Admission_grade': 140,
                'Age_at_enrollment': 19,
                'Curricular_units_1st_sem_grade': 14.2,
                'Curricular_units_2nd_sem_grade': 13.8,
                'Tuition_fees_up_to_date': 1,
                'Debtor': 0,
                'Scholarship_holder': 1,
                'description': 'High-performing student with good financial status'
            },
            {
                'Previous_qualification_grade': 11.2,
                'Admission_grade': 105,
                'Age_at_enrollment': 24,
                'Curricular_units_1st_sem_grade': 9.5,
                'Curricular_units_2nd_sem_grade': 8.8,
                'Tuition_fees_up_to_date': 0,
                'Debtor': 1,
                'Scholarship_holder': 0,
                'description': 'Struggling student with financial difficulties'
            },
            {
                'Previous_qualification_grade': 13.1,
                'Admission_grade': 125,
                'Age_at_enrollment': 21,
                'Curricular_units_1st_sem_grade': 11.8,
                'Curricular_units_2nd_sem_grade': 12.1,
                'Tuition_fees_up_to_date': 1,
                'Debtor': 0,
                'Scholarship_holder': 0,
                'description': 'Average student with stable situation'
            }
        ]
        
        for i, student in enumerate(sample_students, 1):
            description = student.pop('description')
            
            # Add required features with reasonable defaults
            student_complete = {
                'Application_mode': 7,
                'Application_order': 1,
                'Course': 8,
                'International': 0,
                'Curricular_units_1st_sem_credited': 6,
                'Curricular_units_1st_sem_enrolled': 6,
                'Curricular_units_1st_sem_evaluations': 6,
                'Curricular_units_1st_sem_approved': 6,
                'Curricular_units_2nd_sem_credited': 6,
                'Curricular_units_2nd_sem_enrolled': 6,
                'Curricular_units_2nd_sem_evaluations': 6,
                'Curricular_units_2nd_sem_approved': 6,
                'GDP': 1.74,
                'Inflation_rate': 1.4,
                'Unemployment_rate': 7.1,
                'Gender': 1,
                'Displaced': 0,
                'Educational_special_needs': 0,
                'Fathers_qualification': 9,
                'Mothers_qualification': 8,
                'Fathers_occupation': 4,
                'Mothers_occupation': 3,
                'Marital_status': 1,
                'Nationality': 1,
                'Daytime_evening_attendance': 1,
                'Previous_qualification': 1,
                'Curricular_units_1st_sem_without_evaluations': 0,
                'Curricular_units_2nd_sem_without_evaluations': 0,
                **student
            }
            
            # Add engineered features
            student_complete['Total_1st_sem_units'] = student_complete['Curricular_units_1st_sem_enrolled']
            student_complete['Total_2nd_sem_units'] = student_complete['Curricular_units_2nd_sem_enrolled']
            student_complete['Success_rate_1st_sem'] = 1.0
            student_complete['Success_rate_2nd_sem'] = 1.0
            student_complete['Overall_grade_average'] = (student_complete['Curricular_units_1st_sem_grade'] + 
                                                       student_complete['Curricular_units_2nd_sem_grade']) / 2
            student_complete['Financial_stress'] = int(student_complete['Tuition_fees_up_to_date'] == 0 or student_complete['Debtor'] == 1)
            student_complete['Economic_pressure'] = 0
            student_complete['Parent_education_support'] = student_complete['Fathers_qualification'] + student_complete['Mothers_qualification']
            student_complete['Family_stability'] = 1
            student_complete['Mature_student'] = int(student_complete['Age_at_enrollment'] > 22)
            student_complete['Age_risk'] = int(student_complete['Age_at_enrollment'] > 25)
            
            try:
                result = predictor.predict_dropout_risk(student_complete)
                
                print(f"\nStudent {i}: {description}")
                print(f"Dropout Risk: {result['dropout_probability']}%")
                print(f"Risk Level: {result['risk_level']}")
                print(f"Prediction: {result['prediction']}")
                print(f"Model Used: {result['model_used']}")
                
            except Exception as e:
                print(f"Error predicting for student {i}: {e}")
        
        print(f"\n=== Analysis Complete ===")
        print(f"Best Model: {predictor.best_model_name}")
        print(f"All visualizations and reports saved to '{predictor.output_dir}' folder")
        print(f"Check the folder for:")
        print("- Target distribution analysis")
        print("- Academic performance analysis") 
        print("- Financial factors analysis")
        print("- Parent education impact")
        print("- Model performance comparison")
        print("- Feature importance analysis")
        print("- Risk analysis and intervention recommendations")
        print("- Detailed intervention report")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nPlease ensure you have the dataset file and update the file path correctly.")