import os
from flask import Flask, redirect, url_for
from flask_login import LoginManager, current_user
from models import db, User, init_db_with_default_admin
from routes.auth import auth_bp
from routes.main import main_bp

def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret-please-change")
    app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", "sqlite:///app.db")
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    db.init_app(app)

    # Login
    login_manager = LoginManager()
    login_manager.login_view = "auth.login"
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    # Blueprints
    app.register_blueprint(auth_bp, url_prefix="/auth")
    app.register_blueprint(main_bp)

    # Create DB and default admin on first run
    with app.app_context():
        db.create_all()
        init_db_with_default_admin()

    # Redirect root to predictions (will bounce to login if not authed)
    @app.route("/")
    def root():
        if not current_user.is_authenticated:
            return redirect(url_for("auth.login"))
        return redirect(url_for("main.index"))

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
