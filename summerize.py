from flask import Flask,render_template,request
from DataPreprocessing import DataPreprocessing
from Prediction import Prediction
app=Flask(__name__)

@app.route("/")
def index():
    return render_template('summary.html')


@app.route('/text',methods=['GET','POST'])
def input():
    #  getting text from user

    if request.method=='POST':
        text=request.form.get("text_in")
    # creating summary
        proc = DataPreprocessing()
        load_data = proc.load_pickle("TokenizerData")
        predictor = Prediction(load_data)
        encoder_model = predictor.load_model('models/encoder_model.json', 'models/encoder_model_weights.h5')
        decoder_model = predictor.load_model('models/decoder_model.json', 'models/decoder_model_weights.h5')
        summary = predictor.generated_summaries(text, encoder_model, decoder_model)

    return render_template("summary.html",summary=summary)


if '__name__'=='__main__':
    app.run(debug=True)


