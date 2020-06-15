# Import modules
from flask import Flask, render_template, request, url_for, send_file
from analysis import analysis

# Configure app
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
with app.app_context():
    options = analysis.get_options()
    indicators = analysis.get_indicators()
    #analysis.save_datasets()

# Route : Index page
@app.route('/')
def root():
    return render_template('index.html', options=options, indicators=indicators)

# API : Options
@app.route('/api/options')
def api_options():
    return {'success':True, 'options':options}

# API : Indicators
@app.route('/api/indicators')
def api_indicators():
    return {'success':True, 'indicators':indicators}

# API : Trends
@app.route('/api/trends')
def api_trends():
    option = request.args.get('option')
    if not option or option not in options:
        return {'success':False, 'message':'Invalid option'}
    buy, sell = analysis.get_trends(option)
    return {'success':True, 'option':option, 'Buy':buy, 'Sell':sell}

# API : Graphs
@app.route('/api/graphs')
def api_graphs():
    option = request.args.get('option')
    indicator = request.args.get('indicator')
    if not option or option not in options:
        return {'success':False, 'message':'Invalid option'}
    if not indicator or indicator not in indicators:
        return {'success':False, 'message':'Invalid indicator'}
    graph = analysis.get_graphs(option, indicator)
    return {'success':True, 'option':option, 'indicator':indicator, 'graph':url_for('static', filename=graph)}

# Run app
if __name__ is '__main__':
    app.run()