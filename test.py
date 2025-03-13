from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Direct HTML</title>
    </head>
    <body>
        <h1>This is direct HTML output</h1>
        <p>No templates used.</p>
    </body>
    </html>
    """

if __name__ == '__main__':
    app.run(debug=True, port=8080)