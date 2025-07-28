import pandas as pd
import os
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from config import get_cfg_defaults

cfg = get_cfg_defaults()

def fetch_dataset(id=cfg.DATA.DATASET_ID):
    dataset = fetch_ucirepo(id=id)
    print(f"Keys of datasets: {dataset.keys()}")
    print("Detailed keys: \n")
    print(f"\t- Data: {dataset.data.keys()}")
    print(f"\t- Metadata: {dataset.metadata.keys()}")
    print(f"\t- Variables: {dataset.variables.keys()}")
    print(f"Number of Features: {dataset.metadata.num_features}")
    print(f"Number of Instances: {dataset.metadata.num_instances}")
    print(f"Name of features: {dataset.data.headers}")
    print(f"Name of the dataset: {dataset.metadata.name}")
    print("\n")
    name_dataset = dataset.metadata.name.replace(" ", "_")
    df = dataset.data.original
    print(f"Header of df: {df.columns}")
    print(f"Shape of df: {df.shape}")
    return df, name_dataset

def filter_null(df, num_dels=cfg.DATA.NUM_DEL_MISSING_COLS):
    # Find the NaN columns
    # Count NaN values in each column
    nan_counts = df.isnull().sum()
    print(f"Percent NaN values per column: {round(nan_counts/len(df)*100, 1)}")
    # Find the columns with the highest number of NaN values
    columns_to_drop = nan_counts.nlargest(num_dels).index.tolist()
    print(f"\nColumns to drop: {columns_to_drop}")
    # Remove the identified columns
    df = df.drop(columns=columns_to_drop)
    print(f"\nShape of df_train after dropping columns: {df.shape}")
    return df

def find_columns(df):
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    return categorical_cols, numerical_cols

def split_data(df, ratio=cfg.DATA.ORACLE_RATIO, seed = cfg.DATA.SEED, total_points=cfg.DATA.TOTAL_PARTITONS, number_dominate=cfg.DATA.NUMBER_DOMINATE):
    #Split Train - Oracle
    df_train, df_test = train_test_split(df, test_size=ratio, random_state=seed)
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    print(f"Shape of Train set: {df_train.shape}")
    print(f"Shape of Test set: {df_test.shape}")
    value_counts = df_test.y.value_counts()
    print("\nDistribution in Test set:")
    for class_name, count in value_counts.items():
        print(f"\t- Class {class_name} has {count} values")
    #Get Small test set
    df_no = df_test[df_test['y'] == 'no'].sample(n=number_dominate, random_state=seed)
    df_yes = df_test[df_test['y'] == 'yes'].sample(n=total_points-number_dominate, random_state=seed)
    df_sampled = pd.concat([df_yes, df_no])
    # Shuffle the combined dataframe
    df_small = df_sampled.sample(frac=1, random_state=seed)
    # Oracle
    df_oracle = df.drop(df_small.index)
    df_small = df_small.reset_index(drop=True)
    df_oracle = df_oracle.reset_index(drop=True)
    return df_train, df_test, df_small, df_oracle

#Split to X (features) and y(target)
def split_to_X_y(df, type='train', encoder=None):
    X = df.drop(columns='y')
    y = df['y']

    if type == 'train':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        return X, y, label_encoder
    else:
        y = encoder.transform(y)
        return X, y

def save_dataset(df, name_dataset, name_file):
    save_dir = os.path.join(cfg.DATA.SAVE_DIR, name_dataset, f"{cfg.DATA.NUMBER_DOMINATE}_{cfg.DATA.TOTAL_PARTITONS - cfg.DATA.NUMBER_DOMINATE}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    df.to_csv(os.path.join(save_dir, name_file), index=False)
    print(f"Dataset saved in {save_dir}")

def get_dataset(id=cfg.DATA.DATASET_ID, num_dels=cfg.DATA.NUM_DEL_MISSING_COLS, ratio=cfg.DATA.ORACLE_RATIO, seed = cfg.DATA.SEED, total_points=cfg.DATA.TOTAL_PARTITONS, number_dominate=cfg.DATA.NUMBER_DOMINATE):
    df, name_dataset = fetch_dataset(id)
    df = filter_null(df, num_dels)
    
    df_train, df_test, df_small, df_oracle = split_data(df, ratio, seed, total_points, number_dominate)

    save_dataset(df_train, name_dataset, 'df_train.csv')
    save_dataset(df_small, name_dataset, f"df_small.csv")
    save_dataset(df_oracle, name_dataset, f"df_oracle.csv")
    return df_train, df_test, df_small, df_oracle

def load_dataset(file_dir):
    df = pd.read_csv(file_dir)
    print(f"Loaded dataset from {file_dir}")
    return df

def preprocessor_data(num_cols, cat_cols):
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, num_cols),
        ('cat', categorical_pipeline, cat_cols)
    ])
    return preprocessor

if __name__ == "__main__":
    # df, name_dataset = fetch_dataset()
    # df = filter_null(df)
    # save_dataset(df, name_dataset, 'demo.csv')
    df = load_dataset('Datasets/Bank_Marketing/300_200/demo.csv')
    print(df.head(3))