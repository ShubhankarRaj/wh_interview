Start the UVICORN Web Server:
\_> uvicorn user_service.main:app --reload

CREATE USER CURL:
curl -XPOST' 'http://127.0.0.1:8000/users/' -H 'accept: application/json' -H 'Content-Type: application/json
' -d '{
"name": "John Nash",
"email": "john.nahs@gmail.com",
"gender": "mail",
"age": 24
}'

EXPECTED OUTPUT:
{"id":<>,"name":"John Nash","email":"john.nahs@gmail.com","gender":"mail","age":24}

QUERY USERS CURL
curl -X 'GET' 'http://127.0.0.1:8000/users/' -H 'accept: application/json'

QUERY A PARTICULAR USER WITH USER-ID 1
curl -X 'GET' 'http://127.0.0.1:8000/users/1' -H 'accept: application/json'
