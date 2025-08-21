from datetime import datetime, date
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class User(UserMixin, db.Model):
    __tablename__ = "users"
    id          = db.Column(db.Integer, primary_key=True)
    username    = db.Column(db.String(80), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(200), nullable=False)
    is_admin    = db.Column(db.Boolean, default=False)
    created_at  = db.Column(db.DateTime, default=datetime.now)

    def set_password(self, raw: str):
        self.password_hash = generate_password_hash(raw)

    def check_password(self, raw: str) -> bool:
        return check_password_hash(self.password_hash, raw)

class Prediction(db.Model):
    __tablename__ = "predictions"
    id            = db.Column(db.Integer, primary_key=True)
    sport         = db.Column(db.String(20), nullable=False, default="mlb")
    game_date     = db.Column(db.Date, nullable=False, index=True)
    home_team     = db.Column(db.String(80), nullable=False)
    away_team     = db.Column(db.String(80), nullable=False)
    model_name    = db.Column(db.String(120), nullable=True)
    proba_home    = db.Column(db.Float, nullable=False)
    pick          = db.Column(db.String(8), nullable=False)   # "HOME" or "AWAY"
    pick_prob     = db.Column(db.Float, nullable=False)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)

def init_db_with_default_admin():
    """
    Ensure default admin exists: username='mike', password='moore'
    """
    existing = User.query.filter_by(username="mike").first()
    if not existing:
        u = User(username="mike", is_admin=True)
        u.set_password("moore")
        db.session.add(u)
        db.session.commit()
