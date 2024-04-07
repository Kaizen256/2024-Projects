import requests
from flask import Flask, render_template, request

app = Flask(__name__)
# Api key i recieved from openweathermap.org
api_key = "856fae7118e2a17a71450ab3230cec88"

@app.route("/")
def index():
    return render_template("weather.html")

@app.route("/weather", methods=["POST"])
def weather():
    location = request.form.get("location")

    # Asked chat gpt what I could do for cities with spaces. This replaces spaces with the character used for encoding space in urls
    lo_encoded = location.replace(" ", "%20")

    try:
        # api website I used
        result = requests.get(f'http://api.openweathermap.org/data/2.5/weather?q={lo_encoded}&units=metric&appid={api_key}')
        # This next entire part is stuff that I just learned online and through chat gpt. It is to make error handling more secure. Otherwise there would be bugs for when the error code is not found or something goes wrong.
        result.raise_for_status()

        # makes the json result a variable
        weather_data = result.json()
        # Puts that data into my weather.html
        return render_template("weather.html", weather=weather_data)

    except requests.RequestException as e:
        # Handles exceptions like network errors
        error_message = "Failed to retrieve weather data. Please try again later"
        return render_template("weather.html", error_message=error_message)
    except Exception as e:
        # Handles other exceptions
        error_message = "An unexpected error occurred. Please try again later"
        return render_template("weather.html", error_message=error_message)
