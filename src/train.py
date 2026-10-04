import random
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from deap import base, creator, tools, algorithms

from preprocessing import load_and_preprocess_data


DATA_PATH = "data/bus_preference_data.xlsx"


# ============================================================
# 1. Load and preprocess data
# ============================================================

X, y, label_encoders, target_encoder = load_and_preprocess_data(DATA_PATH)


# ============================================================
# 2. Train-test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)


# ============================================================
# 3. Train baseline models
# ============================================================

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
rf_model.fit(X_train, y_train)

xgb_model = XGBClassifier(
    random_state=42,
    eval_metric="logloss"
)
xgb_model.fit(X_train, y_train)


print("\nBaseline models trained successfully!")


# ============================================================
# 4. Genetic Algorithm Feature Selection
# ============================================================

random.seed(42)
np.random.seed(42)

feature_names = list(X_train.columns)
number_of_features = len(feature_names)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


def evaluate_individual(individual):

    selected_indices = [
        i for i, bit in enumerate(individual)
        if bit == 1
    ]

    # At least one feature must be selected
    if len(selected_indices) == 0:
        return (0.0,)

    selected_columns = [
        feature_names[i]
        for i in selected_indices
    ]

    X_selected = X_train[selected_columns]

    model = XGBClassifier(
        random_state=42,
        eval_metric="logloss"
    )

    scores = cross_val_score(
        model,
        X_selected,
        y_train,
        cv=cv,
        scoring="f1"
    )

    return (scores.mean(),)


# Create DEAP classes only once
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))

if not hasattr(creator, "Individual"):
    creator.create(
        "Individual",
        list,
        fitness=creator.FitnessMax
    )


toolbox = base.Toolbox()

toolbox.register(
    "attr_bool",
    random.randint,
    0,
    1
)

toolbox.register(
    "individual",
    tools.initRepeat,
    creator.Individual,
    toolbox.attr_bool,
    n=number_of_features
)

toolbox.register(
    "population",
    tools.initRepeat,
    list,
    toolbox.individual
)

toolbox.register(
    "evaluate",
    evaluate_individual
)

toolbox.register(
    "mate",
    tools.cxTwoPoint
)

toolbox.register(
    "mutate",
    tools.mutFlipBit,
    indpb=0.05
)

toolbox.register(
    "select",
    tools.selTournament,
    tournsize=3
)


population = toolbox.population(n=10)

print("\nRunning Genetic Algorithm...")
print("Features:", number_of_features)
print("Population:", 10)
print("Generations:", 5)


population, logbook = algorithms.eaSimple(
    population,
    toolbox,
    cxpb=0.7,
    mutpb=0.05,
    ngen=5,
    verbose=True
)


# ============================================================
# 5. Get best feature subset
# ============================================================

best_individual = tools.selBest(
    population,
    k=1
)[0]

selected_features = [
    feature
    for feature, bit in zip(feature_names, best_individual)
    if bit == 1
]

print("\nSelected features:")
for feature in selected_features:
    print("-", feature)

print("\nNumber of selected features:", len(selected_features))


# ============================================================
# 6. Train GA-XGBoost using selected features
# ============================================================

X_train_ga = X_train[selected_features]
X_test_ga = X_test[selected_features]

ga_xgb_model = XGBClassifier(
    random_state=42,
    eval_metric="logloss"
)

ga_xgb_model.fit(
    X_train_ga,
    y_train
)

print("\nGA-XGBoost model trained successfully!")

print("\nFinal training complete.")