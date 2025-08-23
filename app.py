# app.py
from __future__ import annotations
import os
from flask import Flask
from flask_login import LoginManager
from model import db, User, init_db_with_default_admin

# blueprints
from routes.main import main_bp
from routes.auth import auth_bp  # if you have an auth blueprint; otherwise remove

def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "devsecret")
    app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///site.db")
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    # init extensions
    db.init_app(app)
    login_manager = LoginManager(app)
    login_manager.login_view = "auth.login"  # or "login" if your route lives on app

    @login_manager.user_loader
    def load_user(uid):
        return User.query.get(int(uid))

    # register blueprints ONCE
    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp)  # remove if you don't have it

    # initial DB setup
    with app.app_context():
        db.create_all()
        init_db_with_default_admin()

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
