from DataPreprocessing import DataPreprocessing
from Prediction import Prediction


def flask_button_click():
    # text = get text from flask text box
    proc = DataPreprocessing()
    load_data = proc.load_pickle("TokenizerData")
    predictor = Prediction(load_data)
    encoder_model = predictor.load_model('models\\encoder_model.json', 'models\\encoder_model_weights.h5')
    decoder_model = predictor.load_model('models\\decoder_model.json', 'models\\decoder_model_weights.h5')
    summary = predictor.generated_summaries(text, encoder_model, decoder_model)

    # now display summary on flask 2nd text box


