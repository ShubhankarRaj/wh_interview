from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

import user_service.crud as crud
import user_service.user_model as user_model
import user_service.user_schema as user_schema
from user_service.database import SessionLocal, engine

user_model.Base.metadata.create_all(bind=engine)


app = FastAPI()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/users/", response_model=user_schema.User)
def create_user(user: user_schema.CreateUser, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    return crud.create_user(db=db, user=user)


@app.get("/users/", response_model=list[user_schema.User])
def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    users = crud.get_users(db, skip=skip, limit=limit)
    return users


@app.get("/users/{user_id}", response_model=user_schema.User)
def read_user(user_id: int, db: Session = Depends(get_db)):
    db_user = crud.get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user

@app.post("/query_users/", response_model=list[user_schema.User])
def query_users(user_query: user_schema.QueryUser, db: Session = Depends(get_db)):
    users = crud.query_users(db, user_query=user_query)
    if not users:
        raise HTTPException(status_code=404, detail="No users found matching the criteria")
    return users

