import os
import sys
# DON'T CHANGE THIS !!!
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, send_from_directory
from flask_cors import CORS
from src.models.user import db # This might be unused if not using user model yet
from src.routes.user import user_bp # This might be unused if not using user routes yet
from src.routes.api_routes import api_bp # Import the new API blueprint
from src.routes.risk_routes import risk_bp # Import the risk assessment blueprint
from src.routes.api_routes import fundamental_bp

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), 'static'))
app.config['SECRET_KEY'] = 'asdf#FGSgvasgf$5$WGT'
# Simplest CORS: allow all origins by default
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)

# Register blueprints
# app.register_blueprint(user_bp, url_prefix='/api') # Keep or remove if not used
app.register_blueprint(api_bp, url_prefix='/api') # Register the new API blueprint
app.register_blueprint(risk_bp) # Register the risk assessment blueprint
app.register_blueprint(fundamental_bp, url_prefix='/api')

# uncomment if you need to use database
# app.config['SQLALCHEMY_DATABASE_URI'] = f"mysql+pymysql://{os.getenv('DB_USERNAME', 'root')}:{os.getenv('DB_PASSWORD', 'password')}@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '3306')}/{os.getenv('DB_NAME', 'mydb')}"
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# db.init_app(app)
# with app.app_context():
#     db.create_all()

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    static_folder_path = app.static_folder
    if static_folder_path is None:
            return "Static folder not configured", 404

    if path != "" and os.path.exists(os.path.join(static_folder_path, path)):
        return send_from_directory(static_folder_path, path)
    else:
        index_path = os.path.join(static_folder_path, 'index.html')
        if os.path.exists(index_path):
            return send_from_directory(static_folder_path, 'index.html')
        else:
            return "index.html not found", 404

@app.route('/cors-test')
def cors_test():
    return {"msg": "CORS test"}

@app.route('/routes')
def list_routes():
    import urllib
    output = []
    for rule in app.url_map.iter_rules():
        methods = ','.join(rule.methods)
        line = urllib.parse.unquote(f"{rule.endpoint}: {rule.rule} [{methods}]")
        output.append(line)
    return "<br>".join(sorted(output))


if __name__ == '__main__':
    # Only use app.run for local development
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
