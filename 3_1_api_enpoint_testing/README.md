This is a supplemntary module which I created for testing FastAPIs and mocking databases.

## SETUP

```
_> cd 3_1_api_endpoint_testing
_> poetry install

# Start the Uvicorn server
_> uvicorn user_service.main:app --reload

```

Create User CURL:

```
curl -XPOST' 'http://127.0.0.1:8000/users/' -H 'accept: application/json' -H 'Content-Type: application/json
' -d '{
"name": "John Nash",
"email": "john.nahs@gmail.com",
"gender": "mail",
"age": 24
}'
```

> Expected Output
>
> > `{"id":<>,"name":"John Nash","email":"john.nahs@gmail.com","gender":"mail","age":24}`

Query All Users:

```QUERY USERS CURL
curl -X 'GET' 'http://127.0.0.1:8000/users/' -H 'accept: application/json'
```

Query a user using user id

```
curl -X 'GET' 'http://127.0.0.1:8000/users/1' -H 'accept: application/json'
```

Query multiple users using just the gender(all male):

curl -X 'POST' 'http://127.0.0.1:8000/query_users/' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{
"gender": "male"
}'
