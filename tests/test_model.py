# tests/test_model.py
import unittest
import pandas as pd
from src.model import scale_features

class TestModel(unittest.TestCase):
    def test_scale_features(self):
        # Crear un DataFrame de ejemplo
        data = {'feature1': [1, 2, 3], 'feature2': [4, 5, 6]}
        df = pd.DataFrame(data)
        
        # Llamada a la función
        scaled_df = scale_features(df)
        
        # Verifica que el resultado sea un DataFrame con los índices correctos
        self.assertIsInstance(scaled_df, pd.DataFrame)
        self.assertEqual(scaled_df.shape, df.shape)

if __name__ == "__main__":
    unittest.main()
