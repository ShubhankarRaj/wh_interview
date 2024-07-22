from pydantic import BaseModel

class User(BaseModel):
    id: int
    name: str
    email: str
    gender: str
    age: int

    class Config:
        orm_mode = True

class CreateUserBase(BaseModel):
    email: str
    
class CreateUser(CreateUserBase):
    name: str
    gender: str
    age: int

class QueryUser(BaseModel):
    id: int = None
    name: str = None
    email: str = None
    gender: str = None
    age: int = None
