import pandas as pd

label_mapping = {'light': 0, 'medium': 1, 'heavy': 2}
reverse_label_mapping = {v: k for k, v in label_mapping.items()}


class Mix:
    def __init__(self, id1, id2, v1, v2, dataset):
        self.id1 = id1
        self.id2 = id2
        self.v1 = v1
        self.v2 = v2
        self.dataset = dataset
        self.sara1 = self.get_sara(id1)
        self.sara2 = self.get_sara(id2)

        self.type1 = reverse_label_mapping[self.generate_type(id1)]
        self.type2 = reverse_label_mapping[self.generate_type(id2)]

        self.CII = 0.7
        self.Svalue = 1
        self.Pvalue = 1
        self.TSI = 1.5
        self.predicted = "Stable"


    def get_sara(self, id):
        df = pd.read_csv(f"./data/{self.dataset}")
        row = df[df['Sample ID'] == id]
        sara_array = row[['S', 'Ar', 'R', 'As']].to_numpy().flatten()

        return sara_array

    def generate_type(self, id):
        import joblib

        clf = joblib.load('./models/random forest_density_group.pkl')

        df = pd.read_csv(f"./data/{self.dataset}")
        row = df[df['Sample ID'] == id]

        features = ['Density', 'S', 'Ar', 'R', 'As']
        X = row[features]

        predicted_type = clf.predict(X)[0]

        return predicted_type
