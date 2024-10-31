# tests/test_data_processing.py
import unittest
from src.data_processing import extract_features

class TestDataProcessing(unittest.TestCase):
    def test_extract_features(self):
        # Prueba con parámetros de ejemplo
        path = "C:/Users/jonat/Desktop/PuertoWilches/Audios/All/SMA03126_20210323_050000.wav"
        start, end = 0, 2
        freq_range = (0, 8000)
        
        # Llamada a la función
        features = extract_features(path, start, end, freq_range=freq_range)
        
        # Verifica que el resultado tenga las claves correctas
        self.assertIsInstance(features, dict)
        self.assertIn('spectral_centroid', features)
        self.assertIn('bandwidth', features)
        self.assertIn('spectral_flatness', features)

if __name__ == "__main__":
    unittest.main()
