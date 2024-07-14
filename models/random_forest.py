from sklearn.ensemble import RandomForestClassifier
from config import N_ESTIMATORS, RANDOM_STATE

RFClassifier = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)

# class RFClassifier(RandomForestClassifier):
#     def __init__(self, n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE):
#         super().__init__(n_estimators, random_state)