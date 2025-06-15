import pandas as pd
import numpy as np
from sklearn.utils.extmath import density

import ui.state as state
import joblib

from utils.markers import asses_cii, asses_p_value, asses_tsi, asses_s_value
from utils.ranking import light_oil_ans, medium_oil_ans, heavy_oil_ans

label_mapping = {'light': 0, 'medium': 1, 'heavy': 2}
stability_mapping = {'stable': 0, 'lower stability': 1, 'unstable': 2}
reverse_label_mapping = {v: k for k, v in label_mapping.items()}


class Mix:
    def __init__(self, id1, id2, v1, v2, data=None):
        self.id1 = id1
        self.id2 = id2
        self.v1 = v1
        self.v2 = v2

        self.dataset = state.prediction_dataset

        self.sample1 = None
        self.sample2 = None
        self.get_samples(data)

        self.sara1 = self.get_sara(id1)
        self.sara2 = self.get_sara(id2)

        self.type1 = reverse_label_mapping[self.predict_type(id1)]
        self.type2 = reverse_label_mapping[self.predict_type(id2)]

        self.CII = self.predict_cii()
        self.Svalue = self.predict_svalue()
        self.Pvalue = self.predict_pvalue()
        self.TSI = self.predict_tsi()

        self.mix_type = self.predict_mix_type()
        self.predicted = self.predict_stability()
        self.mix_type = reverse_label_mapping[self.mix_type]

    def predict_mix_type(self):
        density_1 = self.sample1['Density'].dropna().iloc[0]
        density_2 = self.sample2['Density'].dropna().iloc[0]
        mix_density = (self.v1 * density_1 + self.v2 * density_2)/100
        features = pd.DataFrame([{
            'Density': mix_density,
            'CII': self.CII,
        }])

        model = joblib.load('./models/mlp_density_cii.pkl')

        predicted_type = model.predict(features)[0]

        return predicted_type

    def predict_stability(self):
        if self.mix_type == 'light':
            return light_oil_ans(asses_s_value(self.Svalue), asses_tsi(self.TSI))
        if self.mix_type == 'medium':
            return medium_oil_ans(asses_cii(self.CII), asses_p_value(self.Pvalue), asses_s_value(self.Svalue))
        return heavy_oil_ans(asses_cii(self.CII), asses_p_value(self.Pvalue), asses_s_value(self.Svalue))

    def get_samples(self, data):
        if data is None:
            df = pd.read_csv(f"./data/{self.dataset}")
            self.dataset = state.prediction_dataset
            self.sample1 = df[df['Sample ID'] == self.id1]
            self.sample2 = df[df['Sample ID'] == self.id2]
        else:
            self.sample1 = pd.DataFrame([data[0]])
            self.sample2 = pd.DataFrame([data[1]])
            print(self.sample1)
            print(self.sample2)

    def get_sara(self, id):
        df = pd.read_csv(f"./data/{self.dataset}")
        row = df[df['Sample ID'] == id]

        sara_array = row[['S', 'Ar', 'R', 'As']].to_numpy().flatten()

        return sara_array

    def predict_type(self, id):
        try:
            clf = joblib.load('./models/random forest_density_group.pkl')

            row = self.sample1 if id == self.id1 else self.sample2

            features = ['Density', 'S', 'Ar', 'R', 'As']
            X = row[features]

            predicted_type = clf.predict(X)[0]

            return predicted_type
        except Exception as e:
            print(f"Error predicting type for ID {id}: {e}")
            return "-"

    def predict_cii(self):
        try:
            ensemble = joblib.load("models/asmix_nusvr_ensemble.pkl")


            sample1 = self.sample1[['Density', 'As', 'S', 'R', 'Ar']]
            sample2 = self.sample2[['Density', 'As', 'S', 'R', 'Ar']]

            sample1 = sample1 * self.v1 / 10000
            sample2 = sample2 * self.v2 / 10000

            X = pd.concat([sample1.reset_index(drop=True), sample2.reset_index(drop=True)], axis=1)

            features = [
                'D1_scaled', 'As1_scaled', 'S1_scaled', 'R1_scaled', 'Ar1_scaled',
                'D2_scaled', 'As2_scaled', 'S2_scaled', 'R2_scaled', 'Ar2_scaled',
            ]
            X.columns = features

            preds = np.mean([model.predict(X) for model in ensemble], axis=0)

            return round(preds[0], 5)
        except Exception as e:
            print(f"Error predicting CII for IDs {self.id1} and {self.id2}: {e}")
            return "-"

    def predict_tsi(self):
        try:
            ensemble = joblib.load("models/tsi_value_ensemble6.pkl")

            if pd.isna(self.sample1['TSI'].iloc[0]) or pd.isna(self.sample2['TSI'].iloc[0]):
                return "-"

            sample1 = self.sample1[['TSI', 'Density', 'As']]
            sample2 = self.sample2[['TSI', 'Density', 'As']]

            sample1 = sample1 * self.v1 / 100
            sample2 = sample2 * self.v2 / 100

            As_scaled_mean = (sample1['As'].iloc[0] + sample2['As'].iloc[0]) / 2
            D_scaled_mean = (sample1['Density'].iloc[0] + sample2['Density'].iloc[0]) / 2
            TSI_Value_mean = (sample1['TSI'].iloc[0] + sample2['TSI'].iloc[0]) / 2

            X = pd.DataFrame([[
                TSI_Value_mean,
                As_scaled_mean,
                D_scaled_mean,
            ]], columns=[
                'TSI_Value_mean',
                'As_scaled_mean',
                'D_scaled_mean'
            ])

            preds = np.mean([model.predict(X) for model in ensemble], axis=0)

            return round(preds[0], 5)
        except Exception as e:
            print(f"Error predicting TSI for IDs {self.id1} and {self.id2}: {e}")
            return "-"

    def predict_pvalue(self):
        try:
            ensemble = joblib.load("models/p_value_ensemble_invar.pkl")

            if pd.isna(self.sample1['P_value'].iloc[0]) or pd.isna(self.sample2['P_value'].iloc[0]):
                return "-"

            sample1 = self.sample1[['P_value', 'Density', 'As']]
            sample2 = self.sample2[['P_value', 'Density', 'As']]

            sample1 = sample1 * self.v1
            sample2 = sample2 * self.v2

            P_Value_sum = sample1['P_value'].iloc[0] + sample2['P_value'].iloc[0]
            P_Value_mean = P_Value_sum / 2
            As_scaled_sum = sample1['As'].iloc[0] + sample2['As'].iloc[0]
            As_scaled_mean = As_scaled_sum / 2
            D_scaled_sum = sample1['Density'].iloc[0] + sample2['Density'].iloc[0]
            D_scaled_mean = D_scaled_sum / 2

            X = pd.DataFrame([[
                P_Value_mean, P_Value_sum,
                As_scaled_sum, As_scaled_mean,
                D_scaled_sum, D_scaled_mean
            ]], columns=[
                'P_Value_mean', 'P_Value_sum',
                'As_scaled_sum', 'As_scaled_mean',
                'D_scaled_sum', 'D_scaled_mean'
            ])

            preds = np.mean([model.predict(X) for model in ensemble], axis=0)

            return round(preds[0], 5)
        except Exception as e:
            print(f"Error predicting P-value for IDs {self.id1} and {self.id2}: {e}")
            return "-"

    def predict_svalue(self):
        try:
            ensemble = joblib.load("models/s_value_ensemble_invar.pkl")

            if pd.isna(self.sample1['S_value'].iloc[0]) or pd.isna(self.sample2['S_value'].iloc[0]):
                return "-"

            sample1 = self.sample1[['S_value', 'As']]
            sample2 = self.sample2[['S_value', 'As']]

            sample1 = sample1 * self.v1
            sample2 = sample2 * self.v2

            S_Value_sum = sample1['S_value'].iloc[0] + sample2['S_value'].iloc[0]
            S_Value_mean = S_Value_sum / 2
            As_scaled_mean = (sample1['As'].iloc[0] + sample2['As'].iloc[0]) / 2

            X = pd.DataFrame([[
                S_Value_mean,
                As_scaled_mean,
                S_Value_sum
            ]], columns=[
                'S_Value_mean',
                'As_scaled_mean',
                'S_Value_sum'
            ])

            preds = np.mean([model.predict(X) for model in ensemble], axis=0)

            return round(preds[0], 5)
        except Exception as e:
            print(f"Error predicting S-value for IDs {self.id1} and {self.id2}: {e}")
            return "-"
