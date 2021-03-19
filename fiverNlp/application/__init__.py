from flask import Flask
#from flask_cors import CORS


#cors = CORS()

def create_app():
    """ Creating the Flask app and setting its config """

    #Creating the flask app
    app = Flask(__name__)

    #setting config variables from DevelopmentConfig class in config file
    app.config.from_object('config.Config')

    #Initiaizaing Plugins
    #cors.__init__(app)

    with app.app_context():

        #Incuding Routes
        from . import routes
        routes.tl.start()
        return app
