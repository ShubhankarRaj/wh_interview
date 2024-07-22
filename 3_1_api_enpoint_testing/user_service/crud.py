from sqlalchemy.orm import Session

import user_service.user_model as user_model
import user_service.user_schema as user_schema


def get_user(db: Session, user_id: int):
    return db.query(user_model.User).filter(user_model.User.id == user_id).first()

def get_user_by_email(db: Session, email: str):
    return db.query(user_model.User).filter(user_model.User.email == email).first()    

def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(user_model.User).offset(skip).limit(limit).all()

def create_user(db: Session, user: user_schema.CreateUser):
    db_user = user_model.User(email=user.email, name=user.name, gender=user.gender, age=user.age)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    return db_user

def query_users(db: Session, user_query: user_schema.QueryUser):
    query = db.query(user_model.User)
    
    if user_query.id:
        query = query.filter(user_model.User.id == user_query.id)
    if user_query.name:
        query = query.filter(user_model.User.name == user_query.name)
    if user_query.email:
        query = query.filter(user_model.User.email == user_query.email)
    if user_query.gender:
        query = query.filter(user_model.User.gender == user_query.gender)
    if user_query.age:
        query = query.filter(user_model.User.age == user_query.age)
    
    return query.all()
