from src.predict import Predictor
from dotenv import load_dotenv
predictor = Predictor()
predictor.setup()
results= predictor.predict(audio="f745dded-8189-4038-a528-5dfbb9d55c4b.wav")
print(results)